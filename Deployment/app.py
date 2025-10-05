import streamlit as st
import pandas as pd
import joblib
import wandb
import os
import xgboost as xgb
import yaml

with open("app_config.yaml","r") as file:
    config = yaml.safe_load(file)

wandb_key = config.get("wandb_api_key", "")
wandb_project = config.get("wandb_project", "")
wandb_username  = config.get("wandb_user", "")
cts_cols = config.get("cts_col_names", [])
cat_cols = config.get("cat_col_names", [])
encode_columns = config.get("encode_column_names", [])
data_csv_path = config.get("final_merge_csv_path", "final_merge.csv")

@st.cache_data(show_spinner=True)
def load_final_merge(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Try to parse date columns if present
    for col in ["departure_date", "estimated_arrival"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(how="all")

try:
    final_merge = load_final_merge(data_csv_path)
    if final_merge.empty:
        st.error("Loaded CSV is empty. Please check final_merge_csv_path in app_config.yaml")
except FileNotFoundError:
    st.error(f"Could not find data CSV at: {data_csv_path}. Update app_config.yaml 'final_merge_csv_path'.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data CSV: {e}")
    st.stop()



st.title('Truck Delay Classification')

# Let's assume you have a list of options for your checkboxes
options = ['date_filter', 'truck_id_filter', 'route_id_filter']

# Use radio button to allow the user to select only one option for filtering
selected_option = st.radio("Choose an option:", options)


if selected_option == 'date_filter':
    st.write("### Date Ranges")

    #Date range
    min_date = final_merge['departure_date'].min().date() if 'departure_date' in final_merge else pd.Timestamp.today().date()
    max_date = final_merge['departure_date'].max().date() if 'departure_date' in final_merge else pd.Timestamp.today().date()
    from_date = st.date_input("Enter start date in YYYY-MM-DD : ", value=min_date)
    to_date = st.date_input("Enter end date in YYYY-MM-DD : ", value=max_date)

elif selected_option == 'truck_id_filter':

    st.write("### Truck ID")
    truck_id = st.selectbox('Select truck ID: ', final_merge['truck_id'].unique())
    
elif selected_option=='route_id_filter':
    st.write("### Route ID")
    route_id = st.selectbox('Select route ID: ', final_merge['route_id'].unique())
    
if st.button('Predict'):
    try:
        flag = True

        if selected_option == 'date_filter':
            sentence = "during the chosen date range"
            if 'departure_date' not in final_merge.columns:
                st.error("Column 'departure_date' not found in data.")
                st.stop()
            filter_query = (final_merge['departure_date'] >= pd.to_datetime(str(from_date))) & (final_merge['departure_date'] <= pd.to_datetime(str(to_date)))
        
        elif selected_option == 'truck_id_filter':
            sentence = "for the specified truck ID"
            filter_query = (final_merge['truck_id'] == truck_id)
    
        elif selected_option=='route_id_filter':
            sentence = "for the specified route ID"
            filter_query = (final_merge['route_id'] == str(route_id))
        
        else:
            st.write("Please select at least one filter")
            flag = False
    
        if flag:
            try:
                data = final_merge[filter_query] 
            except Exception as e:
                st.error(f"There is an error occurred while filtering the data with given parameters {e}")

        if data.shape[0]==0 and flag == True:
            st.error("No data was found for the selected filters. Please consider choosing different filter criteria.")
        else:
            try:
                if 'delay' not in data.columns:
                    st.error("Column 'delay' not found in data.")
                    st.stop()
                y_test = data['delay']

                missing_cols = [c for c in (cts_cols+cat_cols) if c not in data.columns]
                if missing_cols:
                    st.error(f"Missing required columns in data: {missing_cols}")
                    st.stop()
                X_test = data[cts_cols+cat_cols].copy()

                truck_data_encoder = joblib.load('truck_data_encoder.pkl')
                truck_data_scaler = joblib.load('truck_data_scaler.pkl')

                encoded_features = list(truck_data_encoder.get_feature_names_out(encode_columns))
                enc_vals = truck_data_encoder.transform(X_test[encode_columns])
                enc_df = pd.DataFrame(enc_vals, columns=encoded_features, index=X_test.index)
                X_test = pd.concat([X_test.drop(columns=encode_columns), enc_df], axis=1)
                X_test[cts_cols] = truck_data_scaler.transform(X_test[cts_cols])
            except Exception as e:
                st.error(f"An error occurred while data preprocessing: {e}")    
            
            else:
                try:
                    os.environ['WANDB_API_KEY'] = wandb_key
                    PROJECT_NAME = wandb_project
                    USER_NAME= wandb_username
                    wandb_run = wandb.init(project=PROJECT_NAME)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                else:
                    try:
                        model_artifact = wandb_run.use_artifact(f"{USER_NAME}/{PROJECT_NAME}/XGBoost:latest", type="model")
                        model_dir = model_artifact.download()
                        wandb_run.finish()
                        model_path = os.path.join(model_dir, "xgb-truck-model.pkl")
                        model = joblib.load(model_dir + "/xgb-truck-model.pkl")
                    except Exception as e:
                        st.error(f"An Error occurred while loading the model from wandb: {e}")
                    else:
                        try:
                            dtest = xgb.DMatrix(X_test, label=y_test)
                            y_preds = model.predict(dtest)
                            X_test['Delay'] = y_preds
                            X_test['truck_id'] = data['truck_id']
                            X_test['route_id'] = data['route_id']
                            result = X_test[['truck_id','route_id','Delay']]
                        except Exception as e:
                            st.error(f"An error occurred during prediction {e}")  
                        else:      
                            if not result.empty:
                                st.write("## Truck Delay prediction "+sentence)
                                st.dataframe(result)
                            else: 
                                st.write(f"Oops!! The model failed to predict anything.")    

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

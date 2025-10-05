import streamlit as st
import pandas as pd
import joblib
import wandb
import os
import xgboost as xgb
import hopsworks
import yaml

# Original app.py backup (pre-CSV change)
with open(os.path.join(os.path.dirname(__file__), "app.py"), "r", encoding="utf-8") as f:
    _ = f.read()


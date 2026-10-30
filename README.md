# Star Classification Project

## Dataset
Download the SDSS star-classification CSV from Kaggle and save it as `star_classification.csv` in this folder. It must include columns `u`, `g`, `r`, `i`, `z`, `redshift`, and `class`.

## Install and run notebook
```bash
pip install -r requirements.txt
jupyter notebook star_classification_eda_models.ipynb
```
Run cells from top to bottom. The notebook performs EDA, preprocessing, trains four models, compares them, and saves pipelines in `saved_models/`.

## Run space-themed frontend
After running the notebook successfully:
```bash
streamlit run app.py
```
The app asks for six measurements and uses the best model selected by Macro F1.

## Notes
- Metrics are generated only after training on your downloaded CSV.
- The notebook checks missing values and imputes missing numeric inputs inside model pipelines if needed.
- Outliers are inspected, not automatically removed.
- The app's default values are placeholders. Use real astronomical measurements for meaningful predictions.

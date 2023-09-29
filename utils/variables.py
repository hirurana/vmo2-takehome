"""Project constants"""
random_state = 42
y_label = "Adopted"
data_url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
object_cols = ["Gender", "Type", "Color1", "Color2", "Sterilized", "Vaccinated"]
ordinal_cols = ["FurLength", "Health", "MaturitySize"]
artifact_output_filepath = "artifacts/model"
predicted_output_path = "output/results.csv"

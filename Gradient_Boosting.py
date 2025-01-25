import numpy as np
import pandas as pd
from sklearn.preprocessing  import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

if __name__ == "__main__":
    catNames = ["Cat", "Lion", "cheetah", "Leopard", "Tiger", "Lion", "Cheetah"]
    df = pd.DataFrame({"catNames": catNames})

    catNamesEncoder = OneHotEncoder()
    catNamesEncoded = catNamesEncoder.fit_transform(df[["catNames"]])

    data = pd.read_csv(r"C:\Users\vivek\Downloads\archive\housing.csv")


    calHousing = pd.read_csv(r"C:\Users\vivek\Downloads\archive\housing.csv")

    numAttribs = list(calHousing)[:-1]
    catAttribs = [list(calHousing)[-1]]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('norm', Normalizer()),
        ('scalar', StandardScaler()),
    ])

    fullPipeline = ColumnTransformer([
        ("num", num_pipeline, numAttribs),
        ("cat", OneHotEncoder(), catAttribs),
    ])

    calHousing = fullPipeline.fit_transform(calHousing)

    pause = 1

import pandas as pd
from impute import impute_drop_missing_values
from scale import robust_scale
from outliers import windsorize

df = pd.read_csv('../CC_GENERAL.csv',index_col='CUST_ID')

df.columns = [col.lower() for col in df.columns]

# impute
df = impute_drop_missing_values(df)

# scale
scaled = robust_scale(df)

# windsorize
windsor = windsorize(scaled)

windsor.to_csv("../preprocessed_data.csv",index = False)
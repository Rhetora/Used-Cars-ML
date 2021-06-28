import pandas as pd
import pickle

full = pd.read_csv('cleaned_cars.csv')

train = full.sample(frac=0.6)

validate = full.drop(train.index).sample(frac=0.5)

test = full.drop(train.index).drop(validate.index)

print("train")
print(train.head())
print(train.shape[0])
print("validate")
print(validate.head())
print(validate.shape[0])
print("test")
print(test.head())
print(test.shape[0])

with open("train_set", 'wb') as f:
        pickle.dump(train, f)

with open("validation_set", 'wb') as f:
        pickle.dump(validate, f)

with open("test_set", 'wb') as f:
        pickle.dump(test, f)

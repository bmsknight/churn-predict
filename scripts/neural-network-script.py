import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

import src.constants as const
from src.nn_utils import ChurnDataset, SimpleFeedForwardNetwork, Learner
from src.utils import load_dataset, Standardizer, Evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training will happen on : ", device)

X, y = load_dataset(const.DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)

# scaler = Standardizer(columns_to_standardize=const.NUMERIC_COLUMNS+const.CATEGORICAL_COLUMNS)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = ChurnDataset(X_train, y_train)
test_dataset = ChurnDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleFeedForwardNetwork(in_features=X.shape[1])
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_trainer = Learner(model, loss_fn, optimizer, 100, device)
history = model_trainer.train(train_loader)
y_pred = model_trainer.predict(test_loader)

results = Evaluation(actuals=y_test, predictions=y_pred)
results.print()
print("Hello")

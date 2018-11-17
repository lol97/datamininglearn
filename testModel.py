import pickle as pkl

model = pkl.load(open('modelSVM.sav','rb'))

print(model.predict([[4328.72,4011.79,4296.41,4155.9,4343.59,4582.56,4097.44,4630.77,4217.44,4235.38,4210.77,4287.69,4632.31,4396.41]]))

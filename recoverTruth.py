#!/usr/bin/env python
import numpy as np
import pandas, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping
from PixelateHistogram import *
from scipy.special import gammainc, gammaincc, erfinv
from scipy.stats import norm
import matplotlib.pyplot as plt
from PixelateHistogram.losses import Significance
from PixelateHistogram.layers import *

def getModel_AE(nBINS):
	input_layer = Input(shape=(nBINS,1))
	yields    = Lambda(lambda x: K.sum(x, axis=1), name="compute_yields")(input_layer)
	yieldsInv = Lambda(lambda x: 1./K.sum(x, axis=1), name="compute_yieldsInv")(input_layer)
	
	input_layer1=Dot(axes=[-1,-1])([input_layer, yieldsInv])
	dense = Reshape((nBINS,))(input_layer1)
	#dense   = Dense(units=nBINS, activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	#dense   = Dense(units=50,    activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	dense   = Dense(units=200,    activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	#dense   = Dense(units=8,     activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	dense   = Dense(units=30,     activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	#dense   = Dense(units=50,     activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	dense   = Dense(units=200,     activation='relu',    kernel_initializer='uniform', bias_initializer='zeros') (dense)
	softmax   = Dense(units=nBINS, activation='softmax', kernel_initializer='uniform', bias_initializer='zeros') (dense)
	#softmax = Softmax() (dense)
	predictions = Reshape((nBINS,1))(softmax)  
	
	predictions_rescaled=Dot(axes=[-1,-1])([predictions, yields])
	
	model = Model(inputs=input_layer, outputs=predictions_rescaled)
	model.compile(loss=Significance, optimizer=RMSprop(lr=1e-4))
	
	modelDir= 'output_AE'
	os.system('mkdir -p '+modelDir)
	
	return model,modelDir
	
	
def getModel_Pixelate(nBINS, n_sigma=0):
	input_layer = Input(shape=(nBINS,1))
	yields = Lambda(lambda x: K.sum(x, axis=1), name="compute_yields")(input_layer)
	
	print ('input_layer', input_layer.shape)
	
	
	pixels = PixelateLayer(stepSize=0.01, n_sigma=n_sigma, flatten=True)(input_layer)
	
	dense   = Dense(units=nBINS, activation='relu', kernel_initializer='uniform', bias_initializer='zeros', name="dense0") (pixels)
	softmax   = Dense(units=nBINS, activation='softmax', kernel_initializer='uniform', bias_initializer='zeros', name="softmax") (dense)
	predictions = Reshape((nBINS,1))(softmax)
	
	predictions_rescaled=Dot(axes=[-1,-1])([predictions, yields])
	
	model = Model(inputs=input_layer, outputs=predictions_rescaled)
	model.compile(loss=Significance, optimizer=RMSprop(lr=1e-4))
	
	modelDir= 'output_Pixelate_'+str(n_sigma)
	os.system('mkdir -p '+modelDir)
	
	return model,modelDir

def getModel_PixelateNoise(nBINS, n_sigma=0):
	input_layer = Input(shape=(nBINS,1))
	input_layer2 = PoissonianNoise()(input_layer)
	yields = Lambda(lambda x: K.sum(x, axis=1), name="compute_yields")(input_layer2)
	
	pixels = PixelateLayer(stepSize=0.01, n_sigma=n_sigma, flatten=True)(input_layer2)
	
	dense   = Dense(units=nBINS, activation='relu', kernel_initializer='uniform', bias_initializer='zeros', name="dense0") (pixels)
	softmax   = Dense(units=nBINS, activation='softmax', kernel_initializer='uniform', bias_initializer='zeros', name="softmax") (dense)
	predictions = Reshape((nBINS,1))(softmax)
	
	predictions_rescaled=Dot(axes=[-1,-1])([predictions, yields])
	
	model = Model(inputs=input_layer, outputs=predictions_rescaled)
	model.compile(loss=Significance, optimizer=RMSprop(lr=1e-4))
	
	modelDir= 'output_PixelateNoise_'+str(n_sigma)
	os.system('mkdir -p '+modelDir)
	
	return model,modelDir
	
def recoverTruth(modelName, dataDirectory):
	BINS = pandas.read_csv(dataDirectory+'/bins.csv',header=None)
	print (BINS)
	
	nBINS=np.size(BINS)-1
	print(nBINS)
	
	if modelName=="AE":       model,modelDir = getModel_AE(nBINS)
	if modelName=="Pixelate_0": model,modelDir = getModel_Pixelate(nBINS)
	if modelName=="Pixelate_1": model,modelDir = getModel_Pixelate(nBINS,1)
	if modelName=="Pixelate_2": model,modelDir = getModel_Pixelate(nBINS,2)
	if modelName=="PixelateNoise_0": model,modelDir = getModel_PixelateNoise(nBINS)
	
	model.build(input_shape=(nBINS,))
	model.summary()
	plot_model(model, to_file=modelDir+'/model.png', show_shapes=True)
	
	Xtrain = pandas.read_csv(dataDirectory+'/data_train.csv',header=None).values  # pseudo-data histograms
	Ytrain = pandas.read_csv(dataDirectory+'/model_train.csv',header=None).values # truth pdf histograms
	Xtest  = pandas.read_csv(dataDirectory+'/data_test.csv',header=None).values   # pseudo-data histograms
	Ytest  = pandas.read_csv(dataDirectory+'/model_test.csv',header=None).values  # truth pdf histograms
	
	if modelName=="PixelateNoise_0": Xtrain=Ytrain # in that case, Poisson noise is added on top of truth, on the fly, only during training
	
	history =model.fit(x=Xtrain, y=Ytrain, validation_data=(Xtest,Ytest), batch_size=128, epochs=1000, verbose=1, callbacks=[CSVLogger(modelDir+'/training.log'), EarlyStopping(patience=3)])
	
	Ypred = model.predict(Xtest)
	
	
	fileName = modelDir+"/result.csv"
	np.savetxt(fileName, Ypred, delimiter=",")
	
	return

def significance(d, b, sigma):
	epsilon=1e-12
	if d>=b:
		poisson = gammainc(d,b) 
		poisson = min(max(poisson, epsilon), 1-epsilon)
		sigma = np.sqrt(2.)*erfinv(1-2*poisson)
	elif d>0 :
		poisson = gammaincc(d,b)
		poisson = min(max(poisson, epsilon), 1-epsilon)
		sigma = -np.sqrt(2.)*erfinv(1-2*poisson)
	else: sigma=0
	return sigma
	
def displayResults(dataDirectory, modelDir, nrows=1000):
	
	BINS = pandas.read_csv(dataDirectory+'/bins.csv',header=None)
	Xtest= pandas.read_csv(dataDirectory+'/data_test.csv',header=None, nrows=nrows).values  # pseudo-data histograms
	Ytest= pandas.read_csv(dataDirectory+'/model_test.csv',header=None, nrows=nrows).values  # truth pdf histograms
	YtestEstimate= pandas.read_csv(modelDir+'/result.csv',header=None, nrows=nrows).values  # infered pdf histograms
	
	#Ytest_sigma= np.sqrt(Ytest)
	signif_data = [[ significance(x, mu, np.sqrt(mu)) for (x, mu) in zip(lineX, lineMu)]for (lineX, lineMu) in zip(Xtest, Ytest)]
	signif_estimate = [[ significance(x, mu, np.sqrt(mu)) for (x, mu) in zip(lineX, lineMu)]for (lineX, lineMu) in zip(YtestEstimate, Ytest)]
	#print (signif_data)
	#print (signif_estimate)
	
	signif_data0=np.reshape(signif_data, (-1))
	signif_estimate0=np.reshape(signif_estimate, (-1))
	
	plt.hist(signif_data0, density=False, bins=100)
	plt.savefig(modelDir+'/residual_data_truth.png')
	plt.close()
	
	plt.hist(signif_estimate0, density=False, bins=100)
	plt.savefig(modelDir+'/residual_estimate_truth.png')
	plt.close()
	
	Range = [-5,5]

	n, bins, patches = plt.hist(signif_estimate0, density=False, bins=50, range=Range, label='Z(output, truth)')
	n, bins, patches = plt.hist(signif_data0,   density=False, bins=bins, range=Range, label='Z(data, truth)', hatch='=', edgecolor='r', facecolor="None", histtype = 'step')
	plt.legend()
	
	
	#(mu, sigma) = norm.fit(signif_estimate0)
	#print(mu,sigma)
	#rms = np.sqrt(np.mean(signif_estimate0**2))
	#print(rms)
	
	
	#y = norm.pdf( bins, mu, sigma)
	#l = plt.plot(bins, y, 'r--', linewidth=2)
	#(mu, sigma) = norm.fit(signif_data0)
	#print('mu',mu,'\tsigma',sigma)
	#rms = np.sqrt(np.mean(signif_data0**2))
	#print('rms',rms)
	
	plt.savefig(modelDir+'/residuals.png')
	plt.savefig(modelDir+'/residuals.pdf')
	plt.close()
	return
	
def displayResults2(dataDirectory, modelDirPx, modelDirAE, nrows=1000):
	
	BINS = pandas.read_csv(dataDirectory+'/bins.csv',header=None)
	Xtest= pandas.read_csv(dataDirectory+'/data_test.csv',header=None, nrows=nrows).values  # pseudo-data histograms
	Ytest= pandas.read_csv(dataDirectory+'/model_test.csv',header=None, nrows=nrows).values  # truth pdf histograms
	YtestEstimatePx= pandas.read_csv(modelDirPx+'/result.csv',header=None, nrows=nrows).values  # infered pdf histograms
	YtestEstimateAE= pandas.read_csv(modelDirAE+'/result.csv',header=None, nrows=nrows).values  # infered pdf histograms
	
	#Ytest_sigma= np.sqrt(Ytest)
	signif_data = [[ significance(x, mu, np.sqrt(mu)) for (x, mu) in zip(lineX, lineMu)]for (lineX, lineMu) in zip(Xtest, Ytest)]
	signif_estimatePx = [[ significance(x, mu, np.sqrt(mu)) for (x, mu) in zip(lineX, lineMu)]for (lineX, lineMu) in zip(YtestEstimatePx, Ytest)]
	signif_estimateAE = [[ significance(x, mu, np.sqrt(mu)) for (x, mu) in zip(lineX, lineMu)]for (lineX, lineMu) in zip(YtestEstimateAE, Ytest)]
	#print (signif_data)
	#print (signif_estimate)
	
	signif_data0=np.reshape(signif_data, (-1))
	signif_estimatePx0=np.reshape(signif_estimatePx, (-1))
	signif_estimateAE0=np.reshape(signif_estimateAE, (-1))
	
	
	Range = [-5,5]

	n, bins, patches = plt.hist(signif_data0,   density=False, bins=50, range=Range, label='Z(data, truth)' , facecolor="blue")
	n, bins, patches = plt.hist(signif_estimatePx0, density=False, bins=50, range=Range, label='Z( pixNN(data), truth)', hatch='=', edgecolor='r', facecolor="None", histtype = 'step')
	n, bins, patches = plt.hist(signif_estimateAE0, density=False, bins=50, range=Range, label='Z( aeNN(data), truth)', hatch='=', edgecolor='g', facecolor="None", histtype = 'step')
	plt.xlim(Range)
	plt.xlabel("Bin-by-bin residuals")

	plt.legend()
	
	
	#(mu, sigma) = norm.fit(signif_estimate0)
	#print(mu,sigma)
	#rms = np.sqrt(np.mean(signif_estimate0**2))
	#print(rms)
	
	
	#y = norm.pdf( bins, mu, sigma)
	#l = plt.plot(bins, y, 'r--', linewidth=2)
	#(mu, sigma) = norm.fit(signif_data0)
	#print('mu',mu,'\tsigma',sigma)
	#rms = np.sqrt(np.mean(signif_data0**2))
	#print('rms',rms)
	
	plt.savefig('residuals.png')
	plt.savefig('residuals.pdf')
	plt.close()
	
	return

#recoverTruth(modelName="AE",       dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/")
recoverTruth(modelName="Pixelate_0", dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/")
#recoverTruth(modelName="Pixelate_1", dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/")
#recoverTruth(modelName="Pixelate_2", dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/")
recoverTruth(modelName="PixelateNoise_0", dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/")


displayResults(dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/", modelDir='output_AE')
displayResults(dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/", modelDir='output_Pixelate_0')

displayResults2(dataDirectory="data/SPL-17b_700-1erfexp-v1-normW0.1-Nevt1e5-dupM1/", modelDirPx='output_Pixelate_0', modelDirAE='output_AE')


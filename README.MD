# Supply Forecast (LSTM) 
The objective here is to train a LSTM model to predict a company's product demands based on their historical data.
As per usual we need to gather the proper data to train our model with which in this case we wrote a CSV generator function including 4 columns; one for date and the rest for costumer demands related to 3 products.
Afterward we create a sequence from the data, build the LSTM model and train it by passing it the sequence and then plot the result for each product.

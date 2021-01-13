n_outputs = 12

# define the model
inputs = Input(shape=(6,2))
x = LSTM(64)(inputs)
x = Flatten()(x)
x = Dense(units=n_outputs)(x)
model = Model(inputs=inputs, outputs=tf.reshape(x,[-1,6,2]))

# print the model
plot_model(model, 'cnn.png', show_shapes=True)

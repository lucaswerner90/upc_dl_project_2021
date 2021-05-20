def train():
	"""
	Sequence of Operations:

        1.Forward pass entire input batch through encoder.
        2. Initialize decoder inputs as SOS_token, and hidden state as the encoder’s final hidden state.
        3. Forward input batch sequence through decoder one time step at a time.
        4. If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
        5. Calculate and accumulate loss.
        6. Perform backpropagation.
        7. Clip gradients.
        8. Update encoder and decoder model parameters.
	"""
	pass

if __name__ == "__main__":
	
	print("""
	Executes the model training.
	Saves the model to a file.
	""")
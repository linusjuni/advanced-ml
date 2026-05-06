In this exercise you will work with a shallow node embedding implemented in the script shallow_embedding.py.
The code loads a graph from a file: This graph is simulated from a shallow embedding model, so that we
know the ground truth probability of each possible link. In this exercise we will fit a shallow embedding
model to the data and see how well we can estimated the ground truth.
Question D.1: Examine and run the code for loading the graph data.
• Understand how the graph is represented as a matrix as well as in the form of a set of index pairs and
target values.
• It can perhaps help to visualize the adjacency matrix.
Question D.2: Examine and run the implementation of the class Shallow.
• Understand how the node embeddings are implemented using torch.nn.Embedding. Look up the docu-
mentation if needed.
• Understand what the forward function computes. What exactly is the role of the variables rx and tx?
Question D.3: Examine and run the code to fit the model. In this version, the loss is computed on the
entire graph (no train/validation split and no mini batching).
• Experiment with different number of max_step.
• Experiment with different embedding dimensions. How does the embedding dimension influence the
training loss?
Question D.4: Modify the code to use a train/validation split.
• Make a random split of the data (each node pair) into a training set (e.g. 80%) and a validation set
(e.g. 20%).
• Modify the code to train on only the training data.
• Write code to compute the loss of the trained model on the validation set.
• Experiment with different embedding dimensions. What is the optimal embedding dimension when
computing the loss on the validation set?
Question D.5: Hand in your predictions:
• Using the train/validation procedure you have implemented (or any other updates, hacks and modifica-
tions) to optimize the model. Compute what you believe is the best possible predicted link probability.
• Using the provided code, save your predictions in a file, link_probabilities.pt, and hand it in on DTU
Learn.
I will compute the ground truth loss on your predictions and lowest generalization loss will be honored as
the class winner

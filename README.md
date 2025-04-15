<h2>My Studies on My n-grams and My Attempt to Build an LLM Model</h2>
<p>
I refer to it as my attempt because my computer is unfortunately too weak to train my model with more than 100 datasets and 5 epochs ;-;.
However, I tried and although it delivers texts that are sometimes out of context, I believe this is due to the limited dataset and the low number of epochs. Despite that, the results exceeded my expectations.
LLMs also have evaluation metrics such as perplexity, which is measured as:
</p>
<p>
<code>perplexity = exp(-1/n * sum(log(P(wᵢ|w<i))) for i = 1 to N)</code>
</p>
<p>
I know that in my code I used the n-grams and transformer model only, I also know that LLM have many more steps besides these to be created ranging from NLP, Self-attention and multi-heads thing that is found in my code on line 84, as I said I did more for study reasons, in addition to self-attention and multi-head, It has the functions of loss, backpropagation, fine-tuning and so on.
</p>

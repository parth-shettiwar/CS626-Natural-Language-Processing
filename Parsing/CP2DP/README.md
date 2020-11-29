##CP to DP conversion##

Code for converting Constituent Parse tree to Dependency Parsre tree  
Percolation algorithm, where head is found in a sub-graph of height 3 from bottom, is used on the Allen constituent parse trees repeatdely until no more such sub graoh can found. A rule based list is made to identify head in a sub-graph. Rules are also written to identify the head-modifier relation. The final dependency graph is shown using Graphviz.   
For comparison, we have shown the original Allen constituent parse tree, our dependency parse graph and Allen dependency parse graph.  
Just write the sentence in last block of the notebook in the vafriable "sen" for which you want the Cp to Dp conversion.  
In the end, I have commented out some sentences of length 3-4 words and length 5-9 words, on which we did our error analysis.  
Also added an example of dependency graph generated from our algo for the sentence "The quick brown fox jumps over the lazy dog" in the same directory.


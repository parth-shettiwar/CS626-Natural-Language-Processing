Run "python3 memmass2.py" 
after running the commands 
"awk -F' ' '{split($3,a,"-");print $1 " " $2 " " a[1];}' /content/test.txt > /content/processed_test.txt"
and 
"awk -F' ' '{split($3,a,"-");print $1 " " $2 " " a[1];}' /content/train.txt > /content/processed_train.txt"


Please make sure that the files are kept in /content/ folder or just replace this to the $pwd in the whole code.

OR 

You could run the python notebook memmass2.ipynb
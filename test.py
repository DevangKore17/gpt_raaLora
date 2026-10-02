import evaluate  
bleu = evaluate.load('bleu')  
print('BLEU:', bleu.compute(predictions=['hello world'], references=['hello world']))  
rouge = evaluate.load('rouge')  
print('ROUGE:', rouge.compute(predictions=['hello world'], references=['hello world']))  

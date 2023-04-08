
from Progrmme_principale import * 
import time

start = time.time()


cdr1 = cdr("tp_algo.txt")
cdr1._open()
facture1 = facture(cdr1.resultat)
facture1.definition()


statistique1 = statistique(cdr1.resultat)
statistique1.stat()



print(time.time() - start)

# temps d'éxecution réel l est égal à 1 ms
# La complexité est de O(n²) car notre algorithme exploite les boucles imbriquées

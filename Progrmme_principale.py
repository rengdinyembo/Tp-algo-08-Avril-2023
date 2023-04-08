

class client:
    def __init__(self, nom, date_naiss, num_tel, facture):
        self.nom = nom
        self.date_naiss = date_naiss
        self.num_tel = num_tel
        self.facture = facture
        
        
        
class cdr:
    def __init__(self, filename):
        self.filename = filename
        self.liste = []
        self.resultat = []
        self.num = 0
        
    def _open(self):
        file = open(self.filename, "r")
        for ligne in file:
            self.liste.append(ligne.strip())
        self.pile_dict()
        
        self.num = self.resultat[0]["appelant"]
        
    def pile_dict(self):
        a = []
        b = ["identifiant de l'appel", 'type call', ' date et heure','appelant', 'appelé', 'durée', 'taxe', 'total volume']
        for i in self.liste:
            a = i.split('|')
            self.resultat.append(dict(zip(b, a)))
            
class facture:
    def __init__(self,liste_dict):
        self.liste_dict = liste_dict
        self.prix = 0
    
    def definition(self):
        for i in self.liste_dict:
            type_call = float(i['type call'])
            num2 = i['appelé']
            if i['durée'] != '':
            	dure = float(i['durée'])
            else:
            	dure = 0
            total_vol = float(i['total volume'])
            taxe = float(i['taxe'])
            prix = 0
            if type_call == 0:
                if num2[0:4] == '24381' or num2[0:4] == '24382':
                    prix += 0.025*dure/60
                else:
                    prix += 0.05*dure/60
            elif type_call == 1:
                if num2[0:4] == '24381' or num2[0:4] == '24382':
                    prix  += 0.001
                else:
                    prix += 0.002
            else:
                prix += total_vol*0.03
            if taxe == 1:
                prix = prix + prix *5/100
            elif taxe == 2:
                prix = prix + prix *16/100
            self.prix += prix
        
            
                
class statistique:
    def __init__(self, liste_dict):
        self.liste_dict = liste_dict
        self.nbr_appel = 0
        self.nbr_sms = 0
        self.giga = 0 
        self.dure_appel = 0
        self.statistique = []
        self.stat_dict = {}
    def stat(self):
        for i in self.liste_dict:
            type_call = float(i['type call'])
            
            if i['durée'] != '':
            	dure = float(i['durée'])
            else:
            	dure = 0
            total_vol = float(i['total volume'])
            
            if type_call == 0:
                self.nbr_appel += 1
                self.dure_appel += dure
                
            elif type_call == 1:
                self.nbr_sms += 1
            else:
                self.giga += total_vol
        
        self.statistique = [self.nbr_appel, self.dure_appel, self.nbr_sms, self.giga]
        self.have_stat()
    def have_stat(self):
        a = ["Nombre d'appels", "Durée d'appel", "Nombre de sms", "Nombre de Gigabytes"]
        self.stat_dict = dict(zip(a, self.statistique))
        
            
            
            


    
    
        
            

        
                
            
        
        
    
    


import numpy as np

class YapaySinirAgi:
    def __init__(self, giris_n, gizli_n, cikis_n, ogrenme_hizi=0.1):
        self.lr = ogrenme_hizi
        
        self.W1 = np.random.randn(giris_n, gizli_n) * np.sqrt(1. / giris_n)
        self.W2 = np.random.randn(gizli_n, cikis_n) * np.sqrt(1. / gizli_n)
        
        self.b1 = np.zeros((1, gizli_n))
        self.b2 = np.zeros((1, cikis_n))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_turev(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def egit(self, x_tek, y_tek):
       
        cikti = self.forward(x_tek)

        hata = cikti - y_tek
        cikti_delta = hata * self.sigmoid_turev(cikti)

        gizli_hatasi = np.dot(cikti_delta, self.W2.T)
        gizli_delta = gizli_hatasi * self.sigmoid_turev(self.a1)

        self.W2 -= self.lr * np.dot(self.a1.T, cikti_delta)
        self.b2 -= self.lr * cikti_delta

        self.W1 -= self.lr * np.dot(x_tek.T, gizli_delta)
        self.b1 -= self.lr * gizli_delta

        loss = np.mean((y_tek - cikti) ** 2)

        return loss
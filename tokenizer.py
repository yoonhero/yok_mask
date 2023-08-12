from konlpy.tag import Okt

class Tokenizer():
    def __init__(self):
        self.okt = Okt()
        self._load_token()
        self.UNK = self.tokens.index("<UNK>")
        print(self.tokens[:100])

    def _load_token(self):
        with open("./token.txt", "r", encoding="utf-8") as f:
            self.tokens = f.read().split("\n")

    def tokenize(self, words:list[str]) -> list[list[int]]:
        indices = []
        for word in words:
            cleaned = self.okt.pos(word)
            temp = []
            for w_ in cleaned:
                w = w_[0]
                if w not in self.tokens:
                    temp.append(self.UNK)
                    continue
                temp.append(self.tokens.index(w))
            indices.append(temp)

        return indices


        

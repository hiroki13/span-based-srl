UNK = u'UNKNOWN'


class Vocab(object):
    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        return self.w2i.get(word)

    def get_id_or_unk_id(self, word):
        if word in self.w2i:
            return self.w2i.get(word)
        return self.w2i.get(UNK)

    def get_and_add_id(self, word):
        self.add_word(word)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def has_key(self, word):
        return word in self.w2i

    def size(self):
        return len(self.i2w)

class Word(object):
    def __init__(self, form, mark, sense, prop):
        self.form = form.lower()
        self.string = form
        self.mark = mark
        self.sense = sense
        self.prop = prop

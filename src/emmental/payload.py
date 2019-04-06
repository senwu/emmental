class Payload(object):
    """An advanced class of dataloaders"""

    def __init__(self, name, dataloader):
        self.name = name

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

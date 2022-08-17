
if __name__ == '__main__':
    from transformers import eBart

    model = eBart.from_pretrained("dalle-mini/dalle-mini")
    model("Programmer in a forest")
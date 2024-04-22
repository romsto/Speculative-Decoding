DATASET = {
    "cnndm": {
        "name": "cnn_dailymail",
        "version": "3.0.0",
        "target": "bbhattar/flan_t5_xl_cnn_dailymail",
        "input": lambda data: data["article"],
        "label": lambda data: data["highlights"],
    },
    "wmt14": {
        "name": "wmt14",
        "version": "de-en",
        "target": "leukas/mt5-large-wmt14-deen",
        "input": lambda data: data["translation"]["de"],
        "label": lambda data: data["translation"]["en"],
    },
}

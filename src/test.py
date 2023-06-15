


def test_starchat():
    from generators import StarChat
    # model_name = "bigcode/starcoder"
    model_name = "HuggingFaceH4/starchat-alpha"  # Language model that finetuned from StarCoder to act as helpful coding asistant
    query = "write a python function that can calculate cosine similarity between two vector"

    code_assistant = StarChat(model_name)
    print(code_assistant)
    print("Used device: ", code_assistant._get_device())
    print("Model response: \n\n")
    print(code_assistant(query))

def test_starcoder():
    from generators import StarCoder
    model_name = "bigcode/starcoder"
    g = StarCoder(model_name)
    print(g('def fibonacci(n):', {'max_new_tokens': 10}))

def test_santacoder():
    from generators import SantaCoder
    g = SantaCoder()
    print(g('def fibonacci(n):', {'max_new_tokens': 60}))


if __name__ == '__main__':
    # test_starcoder()
    test_santacoder()
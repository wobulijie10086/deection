


import jieba
import pickle


class RelevanceChat():
    def __init__(self, topk=5):
        self.topk = topk
        self.fited = False

    def fit(self, x_data, y_data, ws_decode):
        self.dict = {}
        high_fw_max = int(len(x_data) * 0.6)
        for ask, answer in zip(x_data, y_data):
            ask_str = ''.join(ask)
            if len(ask_str) == 0:
                continue
            top_key = jieba.lcut(ask_str)
            # print("top key:", top_key)
            y_code = ws_decode.transform(answer)[0]
            key_set = set(top_key)
            for key in key_set:
                rel_list = []
                if key in self.dict:
                    rel_list = self.dict[key]
                    if rel_list[0] == 0:
                        continue
                    elif len(rel_list) >= high_fw_max:
                        print("key list over:", key, "ask:",
                              ask_str)
                        self.dict[key] = [0]
                        continue
                rel_list.append(y_code)
                self.dict[key] = rel_list


                dict_items = self.dict.items()
                print("size:", len(self.dict))
                print("dict:", dict_items)
                self.fited = True


    def relevance(self, ask, answer):
        assert self.fited, "RelevanceChat 尚未进行 fit 操作"
        top_key = jieba.lcut(''.join(ask))
        # print("top key:", top_key)
        key_set = set(top_key)
        key_size = len(key_set)
        if key_size == 0:
            return 0.0
        rel_num = 0
        high_fw = 0
        for key in key_set:
            rel_list = self.dict.get(key)
            if rel_list is not None:
                if rel_list[0] == 0:
                    high_fw += 1
                elif answer in rel_list:
                    rel_num += 1


            if rel_num == 0:
                relv_val = float(high_fw) / key_size
            else:
                relv_val = float(rel_num) / (key_size - high_fw)
            return relv_val


def test():
    x_data, y_data = pickle.load(open('pkl/chatbot.pkl', 'rb'))
    ws_decode = pickle.load(open('pkl/ws_decode.pkl', 'rb'))
    relv = RelevanceChat(5)
    relv.fit(x_data, y_data, ws_decode)
    count = 0
    for ask, answer in zip(x_data, y_data):
        decode = ws_decode.transform(answer)[0]
    relv_val = relv.relevance(ask, decode)
    if relv_val < 0.7:
        print("rel:", relv_val)
        print("ask:", ''.join(ask))
        print("answer:", ''.join(answer), end='\n\n')
        count += 1

    print("same dialogue Confidence<0.7 count:", count)

    count = 0
    for i, answer in enumerate(y_data):
        decode = ws_decode.transform(answer)[0]
        for j, ask in enumerate(x_data):
            if i == j:
                continue
        relv_val = relv.relevance(ask, decode)
        if relv_val > 0.7:
            # print("rel:", relv_val)
            # print("ask:",''.join(ask))
            # print("answer:", ''.join(answer),end='\n\n')
            count += 1
    print("different dialogue Confidence<0.7 count:", count)


if __name__ == '__main__':
    test()

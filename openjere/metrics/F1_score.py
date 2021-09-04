from typing import Dict, List, Tuple, Set, Optional
import json

# relation_vocab = json.load(open("data/nyt/wdec/relation_vocab.json", "r", encoding="utf-8"))


class F1_triplet(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10
        self.wrong_case = []

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10
        self.wrong_case = []

    def get_metric(self, reset: bool = False, generate_wrongcase=False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}
        if generate_wrongcase:
            json.dump(self.wrong_case, open("wrong_case.json", "w", encoding="utf-8"), ensure_ascii=False)

        return result

    def __call__(
        self,
        predictions: List[List[Dict[str, str]]],
        gold_labels: List[List[Dict[str, str]]],
        text: List[str],
        get_seq=lambda dic: (dic["object"], dic["predicate"], dic["subject"]),
    ):

        for i, (g, p) in enumerate(zip(gold_labels, predictions)):
            for pp in p:
                if pp["object"] == pp["subject"]:
                    p.remove(pp)
                elif pp["object"] not in text[i] or  pp["subject"] not in text[i]:
                    p.remove(pp)
                # elif pp["predicate"] not in relation_vocab:
                #     p.remove(pp)
            g_set = set("_".join(get_seq(gg)) for gg in g)
            p_set = set("_".join(get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)
            if len(g_set & p_set) != len(g_set):
                self.wrong_case.append({"text":text[i], "gold":g, "predict":p})


class F1_op(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(
        self,
        predictions: List[List[Dict[str, str]]],
        gold_labels: List[List[Dict[str, str]]],
        get_seq=lambda dic: (dic["object"], dic["predicate"]),
    ):

        for g, p in zip(gold_labels, predictions):
            g_set = set("_".join(get_seq(gg)) for gg in g)
            p_set = set("_".join(get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)


class F1_os(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(
        self,
        predictions: List[List[Dict[str, str]]],
        gold_labels: List[List[Dict[str, str]]],
        get_seq=lambda dic: (dic["object"], dic["subject"]),
    ):

        for g, p in zip(gold_labels, predictions):
            g_set = set("_".join(get_seq(gg)) for gg in g)
            p_set = set("_".join(get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)


class F1_ps(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(
        self,
        predictions: List[List[Dict[str, str]]],
        gold_labels: List[List[Dict[str, str]]],
        get_seq=lambda dic: (dic["predicate"], dic["subject"]),
    ):

        for g, p in zip(gold_labels, predictions):
            g_set = set("_".join(get_seq(gg)) for gg in g)
            p_set = set("_".join(get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)

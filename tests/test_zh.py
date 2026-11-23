"""Offline checks for the Mandarin filter, prompts, delay line, and overdub mix.

Run: uv run --with av --with numpy python tests/test_zh.py
"""
import os
import queue
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import kodub_llm as k


def test_lang_filter():
    assert k.accept_transcript("zh", "这波分矿开得很快", 0.1, "zh")
    assert k.accept_transcript("chinese", "剑圣先手开了大招", 0.2, "zh")
    assert not k.accept_transcript("en", "nice creep route", 0.1, "zh")
    assert not k.accept_transcript("zh", "GG", 0.1, "zh")
    assert not k.accept_transcript("zh", "这波分矿开得很快", 0.9, "zh")
    assert not k.accept_transcript("ko", "소인 멀티 갔어", 0.1, "zh")
    line = "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"
    assert k.is_junk(line, "zh")
    assert not k.accept_transcript("zh", line, 0.1, "zh")
    assert k.is_junk("字幕由 Amara.org 社群提供", "zh")
    assert k.is_junk("谢谢观看", "zh")
    assert not k.is_junk("这波分矿开得很快", "zh")
    assert k.is_junk("一位玩家的陆队正遭到攻击", "zh")
    assert not k.accept_transcript("zh", "一位玩家的陆队正遭到攻击", 0.1, "zh")
    assert k.is_junk("优优独播剧场——YoYo Television Series Exclusive", "zh")
    assert not k.is_junk("这没法练了，剑圣换边", "zh")
    # Korean path: same junk substrings, Hangul gate, default language.
    assert k.accept_transcript("ko", "소인 멀티 갔고 마킹이야", 0.1, "ko")
    assert not k.accept_transcript("ko", "시청해주셔서 감사합니다", 0.1, "ko")
    assert not k.accept_transcript("en", "hello there", 0.1, "ko")
    assert not k.accept_transcript("zh", "这波分矿", 0.1, "ko")


def test_hallucination():
    assert k.is_hallucination("워크래프트3 중계 시작합니다", "ko")
    assert k.is_hallucination("소인, 영섭, 영삼, 032, TH, 파시오", "ko")
    assert not k.is_hallucination("소인 멀티 갔고 마킹 레벨 3이야", "ko")
    assert k.is_hallucination("네 네 네 네 네 네", "ko")
    assert k.is_hallucination("魔兽争霸3解说. Infi, Fly100%, TH000, Lyn", "zh")
    assert k.is_hallucination("Infi, Fly100%, TH000, Lyn, Moon, 120, LawLiet", "zh")
    assert not k.is_hallucination("Infi这波分矿开得很快，剑圣先手", "zh")
    assert k.is_hallucination("谢谢谢谢谢谢谢谢", "zh")
    assert k.is_hallucination("", "zh")
    assert k.is_hallucination("   ", "ko")


def test_prompts_and_context():
    zh = k.ZH_SYSTEM
    ko = k.KO_SYSTEM
    for name in ("Infi", "Fly100%", "TH000", "Lyn", "Moon", "Fortitude", "LawLiet",
                 "Sok", "Colorful", "YouTiao", "eer0", "LioN", "120", "Happy"):
        assert name in zh, name
        assert name in k.ZH_ASR_PROMPT, name
    for term in ("分矿", "二本", "遗忘骑", "油条", "人皇", "女猎", "[skip]", "fragmentary"):
        assert term in zh, term
    assert "Forsaken Paladin" in zh
    assert "Warcraft 3" in zh
    for term in ("练级", "这没法练了", "黄皮", "小典", "腹地", "Hinterlands", "被网", "拉怪", "偷矿", "wait"):
        assert term in zh, term
    assert k.clean_spoken_english("He can't creep here. Wait, no. He's expanding.") == "He can't creep here."
    assert "wait" not in k.clean_spoken_english("Switching to the side camps, wait, no, the middle.").lower()
    for term in ("포세이큰", "코셀빵", "투생템", "[skip]", "fragmentary", "previous lines"):
        assert term in ko, term
    # --context and the stream title are appended on a copy, not the shared prompt.
    tr = k.LLMTranslator("x", "m", system=zh, extra_context="Fly100% vs Lyn, Bo3")
    assert "Fly100% vs Lyn, Bo3" in tr.system
    assert "Fly100% vs Lyn" not in k.LANGS["zh"]["system"]
    tr.add_stream_context("魔兽争霸3 大帝直播", "大帝LioN")
    assert "大帝LioN" in tr.system and "魔兽争霸3 大帝直播" in tr.system
    assert "大帝LioN" not in k.LANGS["zh"]["system"]
    tr.add_stream_context("other", "nope")
    assert tr.system.count("Streamer:") == 1
    assert k.is_skip_translation("[skip]")
    assert k.is_skip_translation("  [SKIP]. ")
    assert k.is_skip_translation("")
    assert not k.is_skip_translation("He is expanding.")
    assert k.google_url("分矿", "zh-CN").split("&q=")[0].endswith("sl=zh-CN&tl=en&dt=t")
    assert "sl=ko&tl=en" in k.google_url("멀티", "ko")
    assert "langpair=zh-CN%7Cen" in k.mymemory_url("分矿", "zh-CN|en") or "langpair=zh-CN|en" in k.mymemory_url("分矿", "zh-CN|en")


def test_clock_delay_and_mix():
    class Fake:
        def __init__(self):
            self.t = 1000.0
        def __call__(self):
            return self.t

    fake = Fake()
    clock = k.SampleClock(now=fake)
    e1 = clock.stamp(16000)
    assert abs(e1 - 1000) < 1e-6
    fake.t = 1000.001  # burst, faster than realtime
    e2 = clock.stamp(16000)
    assert abs(e2 - 1001) < 1e-3
    fake.t = 1010  # stall
    e3 = clock.stamp(16000)
    assert abs(e3 - 1010) < 1e-3

    line = k.DelayLine(2.0)
    line.push(1000.0, np.ones(16000, np.float32))  # occupies [999, 1000)
    assert len(line.pull_due(1001.0)) == 0
    half = line.pull_due(1001.5)
    assert abs(len(half) - 8000) <= 1
    rest = line.pull_due(1002.0)
    assert abs(len(half) + len(rest) - 16000) <= 1

    marks = [(0, 10.0), (16000, 11.0)]
    assert abs(k.time_of(marks, 8000) - 10.5) < 1e-9

    sr = 16000
    dev = np.ones(sr, np.float32)
    dubs = [{"at": 0.0, "pcm": np.full(8000, 0.5, np.float32), "cur": 0}]
    out = k.mix_block(dev, 0.0, sr, dubs, 0.25)
    assert dubs[0]["cur"] == 8000
    assert 0.7 < float(out[4000]) < 0.85
    assert float(out[12000]) > 0.9
    future = [{"at": 10.0, "pcm": np.ones(100, np.float32), "cur": 0}]
    untouched = k.mix_block(np.ones(1000, np.float32), 0.0, sr, future, 0.2)
    assert future[0]["cur"] == 0
    assert float(np.min(untouched)) > 0.99


def test_sniff_and_cli():
    class Mem:
        def __init__(self, data):
            self.data = data
            self.i = 0
        def read(self, n=-1):
            if n is None or n < 0:
                n = len(self.data)
            b = self.data[self.i:self.i + n]
            self.i += len(b)
            return b

    fmt, replay = k.sniff_container(k._FD(Mem(b"FLV\x01rest-of-flv")))
    assert fmt == "flv"
    assert replay.read(8) == b"FLV\x01rest"
    fmt, replay = k.sniff_container(k._FD(Mem(b"\x47\x00\x11\x22mpegts")))
    assert fmt == "mpegts"
    assert replay.read(4) == b"\x47\x00\x11\x22"

    p = k.build_parser()
    a = p.parse_args(["https://www.douyu.com/122024", "--lang", "zh", "--context", "Fly100% vs Lyn, Bo3", "--overdub"])
    assert a.lang == "zh" and a.overdub and a.context == "Fly100% vs Lyn, Bo3"
    assert a.overdub_delay == 7.0 and a.duck == 0.22
    b = p.parse_args(["https://play.sooplive.com/x/1"])
    assert b.lang == "ko" and not b.overdub and b.context == ""


def test_backlog_and_voice():
    class Q:
        def __init__(self, items):
            self.items = list(items)
        def get_nowait(self):
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)

    import queue as queue_mod
    now = 1000.0
    stale = (now - 20, None, None)
    older = (now - 3, "old", None)
    newer = (now - 0.4, "new", None)
    got, dropped = k.take_live_segment(stale, Q([older, newer]), 8, now=now)
    assert dropped == 2 and got[1] == "new"
    none, dropped = k.take_live_segment(stale, Q([(now - 30, "x", None)]), 8, now=now)
    assert none is None and dropped == 2

    calls = {"edge": 0, "eleven": 0}

    def edge(text, rate):
        calls["edge"] += 1
        raise RuntimeError("WSServerHandshakeError")

    def eleven(text, rate=None):
        calls["eleven"] += 1
        return "pcm"

    voice = k.Voice(edge=edge, eleven=eleven, say=None, fail_limit=2)
    assert voice("a") == "pcm" and calls["edge"] == 1 and calls["eleven"] == 1
    assert voice("b") == "pcm" and calls["edge"] == 2
    assert voice("c") == "pcm" and calls["edge"] == 2 and calls["eleven"] == 3

    # Replay of the Mac timings: a line every 3s, each one 1.55s ASR + 0.8s LLM + 1.65s TTS.
    # The old loop speaks every line in order, so the lag climbs. The new loop keeps the
    # newest line and overlaps TTS with the next ASR, so the lag stays one cycle.
    def climb(n=30, interval=3.0, asr=1.55, llm=0.8, tts=1.65, max_lag=12.0, newest=False, overlap=False):
        ends = [i * interval for i in range(n)]
        t = 0.0
        tts_free = 0.0
        ready = []
        i = 0
        while i < n:
            if t < ends[i]:
                t = ends[i]
            avail = [e for e in ends[i:] if e <= t]
            if not avail:
                t = ends[i]
                continue
            live = [e for e in avail if t - e <= max_lag]
            if not live:
                i += len(avail)
                continue
            chosen = max(live) if newest else live[0]
            i = ends.index(chosen) + 1
            t += asr + llm
            if overlap:
                audio = max(t, tts_free) + tts
                tts_free = audio
            else:
                t += tts
                audio = t
            ready.append(audio - chosen)
        return ready

    old = climb()
    new = climb(newest=True, overlap=True, max_lag=8.0)
    assert max(new) < max(old)
    assert max(new) < 8
    print(f"sim old median {sorted(old)[len(old)//2]:.1f}s max {max(old):.1f}s n={len(old)}")
    print(f"sim new median {sorted(new)[len(new)//2]:.1f}s max {max(new):.1f}s n={len(new)}")


if __name__ == "__main__":
    test_lang_filter()
    test_hallucination()
    test_prompts_and_context()
    test_clock_delay_and_mix()
    test_sniff_and_cli()
    test_backlog_and_voice()
    print("ok")

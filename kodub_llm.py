# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "streamlink>=8.5",
#   "av",
#   "numpy",
#   "sounddevice",
#   "silero-vad",
#   "edge-tts>=7.3.1",
#   "mlx-whisper; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "faster-whisper; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""kodub: live Korean or Mandarin -> English voice dub for streamlink streams.

Korean (SOOP/AfreecaTV and anything else streamlink plays as MPEG-TS) is the default.
Mandarin Warcraft 3 on Douyu, Huya and Douyin uses --lang zh. Streamlink 8.5+ includes
those plugins; their media is FLV, so the opener sniffs the container instead of
assuming MPEG-TS.

Only speech recognised as the chosen source language is dubbed. English speech and
non-speech game audio are skipped.
"""
import argparse, asyncio, io, platform, queue, re, sys, threading, time
from collections import deque

import av
import numpy as np

SR = 16000
HANGUL = re.compile(r"[\uac00-\ud7a3]")
HAN = re.compile(r"[\u4e00-\u9fff]")

# Korean Whisper hallucinations on music/silence. Kept as substrings, same as before.
KO_JUNK = ["시청해주셔서", "구독과 좋아요", "MBC 뉴스", "자막", "감사합니다.", "다음 영상에서"]
# Mandarin hallucinations Whisper emits on silence, mostly subtitle-credit lines from its training data.
ZH_JUNK_RES = [
    re.compile(r"字幕由.{0,40}提供"),
    re.compile(r"请不吝点赞"),
    re.compile(r"订阅[,，、\s]*转发"),
    re.compile(r"转发[,，、\s]*打赏"),
    re.compile(r"明镜与点点"),
    re.compile(r"点点栏目"),
    re.compile(r"感谢您的?(观看|收看)"),
    re.compile(r"谢谢观看"),
    re.compile(r"感谢收看"),
    re.compile(r"下期再见"),
    re.compile(r"本期视频"),
    re.compile(r"字幕组"),
    re.compile(r"中文字幕"),
    re.compile(r"优优独播"),
    re.compile(r"Amara\.org", re.I),
    re.compile(r"我的频道"),
    re.compile(r"欢迎订阅"),
    re.compile(r"打赏支持"),
    re.compile(r"支持明镜"),
    re.compile(r"优优独播剧场"),
    re.compile(r"独播剧场"),
    re.compile(r"YoYo Television", re.I),
    # Warcraft 3's own Mandarin announcer, not the caster. "陆队" is the usual mishear of 部队.
    re.compile(r"正遭到攻击"),
    re.compile(r"(你的|一位玩家的).{0,8}(部队|陆队|基地|工人|建筑)"),
    re.compile(r"需要更多的"),
    re.compile(r"不能在那里建造"),
]

KO_ASR_PROMPT = "워크래프트3 중계. 소인, 영섭, 영삼, 032, TH, 파시오, 예진. 마킹, 코셀빵, 포세이큰, 투생템, 멀티, 런버밀, 헤드헌터, 프리스트, 그런트, 힐링웨이브."
# Short on purpose: Whisper's initial prompt is only ~224 tokens, and a long prompt gets echoed back.
ZH_ASR_PROMPT = (
    "魔兽争霸3解说. Infi, Fly100%, TH000, Lyn, Moon, 120, LawLiet, Colorful, YouTiao, 油条, eer0, LioN, Happy, Fortitude, Sok. "
    "人族, 兽族, 亡灵, 暗夜, 大法师, 山丘, 剑圣, 死骑, 恶魔猎手, 小鹿, 月女, 遗忘骑, 分矿, 偷矿, 练级, 拉怪, 黄皮."
)

KO_SYSTEM = """You translate live Korean speech-to-text of a Warcraft 3 or StarCraft esports caster into English, line by line, for text-to-speech.
The input is automatic speech recognition of a caster calling a pro or semi-pro match. Lines are fast, fragmentary, and full of slang, player nicknames, unit and hero abbreviations, and recognition errors.
The speaker is commentating on other players' games. Players are usually referred to by nickname (e.g. 소인, 영삼/영섭, 032, TH, 파시오).
If a stream title, streamer name, or match context is included below, trust it for who is playing and what kind of match this is.

Rules:
- Use the glossary and the previous lines to infer what the caster meant. Do not translate garbled words literally.
- Write it the way an English Warcraft 3 caster would say it out loud: short and natural. Output ONLY that English line, no quotes, notes, or alternatives.
- Keep player nicknames romanized and consistent with earlier lines. Never turn a nickname into an ordinary English word. Render game terms the way English players say them (unit and building names, "tier 2", "expansion", "creeping", "harass") and keep slang natural (e.g. 투생템 = going straight to tier 2 / skipping items, 멀티 = expansion, 견제 = harass).
- Refer to players as he/they. Use "I" only when the caster clearly talks about himself (his own stream, plans, opinions).
- If the line is pure noise, a subtitle credit, music, or not commentary, output exactly [skip] and nothing else. Do not guess.
- If a line is fragmentary but has a plausible casting meaning, translate that meaning instead of skipping.
- Never second-guess yourself out loud. No "wait", "no", "sorry", or a second version of the line.

Warcraft 3 glossary: 포세이큰/포세이크 = Forsaken (a newly added race, so new Forsaken unit/hero names may appear; "he" when it means that player),
  마킹/MK = Mountain King, 코셀빵 (ASR variants: 코셀, 코셀팡, 코셀방) = Forsaken Paladin (a newly added hero), 휴먼 = Human, 오크 = Orc, 언데드 = Undead, 나엘 = Night Elf,
  팔라 = Paladin, 아크메이지/아크 = Archmage, 블마 = Blademaster, 데나 = Death Knight, 헤드헌터 = Headhunters, 그런트 = Grunts, 라이플맨 = Riflemen,
  프리스트 = Priests, 생추 = Sanctuary, 캐슬 = Castle, 공방 업 = attack/armor upgrades (1-1 = level 1 both), 렙/레벨 = hero level, 3,3렙 = both heroes level 3,
  땡X = going straight/mass X, 사냥 = creeping, 뒷방 = backdoor/base raid, 멀티 = expansion, 본진 = main base, 견제 = harass, 투생템 = tier-2 rush."""

ZH_SYSTEM = """You translate live Mandarin speech-to-text of a Warcraft 3 esports caster into English, line by line, for text-to-speech.
The input is automatic speech recognition of a caster calling a pro or semi-pro match on a Chinese stream (Douyu, Huya, Douyin). Lines are fast, fragmentary, and full of slang, player nicknames, unit and hero abbreviations, and recognition errors.
If a stream title, streamer name, or match context is included below, trust it for who is playing and what kind of match this is.

Rules:
- Use the glossary and the previous lines to infer what the caster meant. Do not translate garbled words literally.
- Write it the way an English Warcraft 3 caster would say it out loud: short and natural. Output ONLY that English line, no quotes, notes, or alternatives.
- Keep player names as their known English handles. Never turn a nickname into an ordinary English word (油条 is the player YouTiao, not fried dough; 彩色 is the player Colorful when it is a name; Moon is the player, 月女 is the hero).
- Refer to players as he/they. Use "I" only when the caster clearly talks about himself.
- If the line is pure noise, a subtitle credit, a TV promo (优优独播剧场), the in-game announcer (forces under attack, not enough gold), music, or not commentary, output exactly [skip] and nothing else. Do not guess.
- If a line is fragmentary but has a plausible casting meaning, translate that meaning instead of skipping.
- Never second-guess yourself out loud. No "wait", "no", "sorry", or a second version of the line. One spoken line only.

Player handles: Infi, Fly100% (also Fly, 百分百), TH000 (also TH), Lyn, Moon, Fortitude, LawLiet, Sok, Colorful (彩色), YouTiao (油条), eer0 (also 一零), LioN (also 大帝 when that is the caster), 120 (also 人皇), Happy, LabyRinth. 浪漫 and 毕竟罗曼蒂 are Fortitude's ladder name, not the adjective "romantic". 罗曼蒂克 without 毕竟 can be the player Romantic. 民主 is a misheard player name, not the word democracy: transliterate it or use a handle already in the match. Other names you recognize (TeD, HawK, Yumiko, Focus, Remind) stay as those handles.

Warcraft 3 glossary, as Chinese casters say it:
- Races: 人族/人 = Human, 兽族/兽 = Orc, 亡灵/不死 = Undead, 暗夜/精灵 = Night Elf.
- Heroes: 大法/大法师 = Archmage, 山丘 = Mountain King, 圣骑 = Paladin, 血法 = Blood Mage, 剑圣 = Blademaster, 先知 = Far Seer, 牛头/酋长 = Tauren Chieftain, 暗猎/暗影猎手 = Shadow Hunter, 死骑/死亡骑士 = Death Knight, 巫妖 = Lich, 恐惧 = Dreadlord, 地穴 = Crypt Lord, 恶猎/恶魔猎手 = Demon Hunter, 守望 = Warden, 月女/月骑 = Priestess of the Moon, 小鹿/KOG = Keeper of the Grove, 熊猫 = Pandaren Brewmaster, 炼金 = Alchemist, 火领 = Firelord, 修补 = Tinker, 娜迦 = Naga Sea Witch, 被遗忘者圣骑士/遗忘骑 = Forsaken Paladin (the new tavern hero).
- Units: 火枪 = Riflemen, 破法 = Spell Breakers, 牧师 = Priests, 女巫 = Sorceress, 民兵 = Militia, 农民 = Peasants, 苦工 = Peons, 大G = Grunts, 猎头 = Headhunters, 狂战 = Berserkers, 狼骑 = Raiders, 蝙蝠 = Batriders, 牛 = Tauren, 飞龙 = Wyverns, 食尸鬼 = Ghouls, 蜘蛛 = Crypt Fiends, 雕像/石像鬼 = Gargoyles, 胖子 = Abomination (the unit, not a hero), 冰龙 = Frost Wyrms, 女妖 = Banshees, 女猎 = Huntresses, 弓箭 = Archers, 树妖 = Dryads, 奇美拉 = Chimeras, 大山 = Mountain Giants, 角鹰 = Hippogryphs, 毁灭 = Destroyer (the unit 毁灭者, not the abstract word "destruction"), 飞机 = flying machine / gyrocopter, not an airplane.
- Buildings and tech: 基地 = main, 兵营 = barracks, 祭坛 = altar, 二本 = tier 2, 三本 = tier 3, 升本 = teching up, 科技 = tech, 塔/箭塔/炮塔 = tower, 商店 = shop, 眼/守卫 = ward.
- Creeping: 练/练级/练怪 = creeping, leveling the hero on neutral creeps. It does not mean practice or training. 这没法练了 = he can't creep here anymore. 这里不练了 = not creeping here anymore. 转而练边上 / 练边上 = switching to creep the side camps. 拉怪 = pulling creeps. 刷野/打野 = creeping too.
- Slang: 分矿/开矿/二矿 = taking an expansion, 主矿 = main gold, 单矿 = one-base, 偷矿 = sniping or denying that expansion, 偷家/换家 = base trade, 运营 = macro, 一波 = all-in, 打架/团战 = fight, 抓 = picking a unit or hero off, 秒 = bursting something down instantly, 网/被网 = netted or ensnared (crypt fiend web or raider net), 骚扰/牵制 = harass, 微操 = micro, 卡兵 = body block, 集火 = focus, 风筝 = kite, 残血 = low HP, 人口 = supply, 爆兵 = a timing push, 攻防/1-1 = attack and armor upgrades, 三级 = hero level 3, 三三 = both heroes level 3, 买活 = buying a revive, 回城/TP = town portal, 圣盾 = Divine Shield, 无敌 = invulnerability potion, 隐身 = invisibility potion, 爪子 = claws, 鞋子 = boots, 圆环 = circlet, 翻盘 = a comeback, 后期 = late game, 开局 = the opening.
- 狗/狗子/小狗 is not always spirit wolves. Casters use it loosely. Obey the race note added under this prompt for the current line: ghouls when Undead is on screen and Orc is not; spirit wolves only when Orc or a Far Seer is on screen and Undead is not; if both races are on screen, ghouls when the line is about the Undead army and wolves only when it is the Far Seer's wolves; if neither race is on screen, say "dog" and do not invent a unit. 切狗 means focus or kill those units, not "switch the build into wolves".
- A lone Latin letter is a hotkey. 换你的C / C过以后 is Death Coil (死亡缠绕) or that hero's C hotkey, not Crypt Fiends, unless the line actually says 蜘蛛.
- Unsure nicknames, do not invent a unit. Keep a short literal or a transliteration. 黑胖 might be a Pit Lord or an Abomination; say which only when the line makes it obvious, otherwise keep 黑胖. 三球 is three orbs on a hero; call it Destroyers only when the line is already about 毁灭者, otherwise say "three orbs".
- 小炮 = Mortar Teams. 黄皮 = the yellow shop scroll (say "scroll", never "yellow skin"). 小典/小电 = a hero skill, usually lightning (Chain Lightning) or, on an Archmage, Brilliance Aura or Blizzard; not a player's name. 腹地 is not the Hinterlands. It is a mishear of 伏击 (an ambush), 附近 (nearby), or 墓地 (the graveyard). Pick from the line. Never say Hinterlands.
- Short map codes (EI, TR, TM, AZ, TS, CH, and similar) are map names; keep the code."""


# Race words used to decide ambiguous slang. 狗 itself is not a marker.
_UD = ("亡灵", "不死族", "食尸鬼", "胖子", "憎恶", "毁灭者", "死骑", "死亡骑士", "巫妖",
       "地穴领主", "地穴", "恐惧魔王", "女妖", "冰龙", "绞肉", "雕像", "石像鬼", "小蜘蛛", "蜘蛛", "undead", "ghoul", "abomination", "destroyer")
_ORC = ("兽族", "兽人", "剑圣", "牛头人", "牛头", "暗影猎手", "暗猎", "大G", "狼骑", "蝙蝠骑", "灵魂行者", "orc", "blademaster")
_FS = ("先知", "远见者", "far seer", "farseer")
_HU = ("人族", "山丘", "大法师", "大法", "圣骑", "圣骑士", "血法", "飞机", "火枪", "破法", "民兵", "狮鹫", "human", "mountain king", "paladin", "gyrocopter")
_NE = ("暗夜", "小鹿", "月女", "恶魔猎手", "恶猎", "守望者", "女猎", "树妖", "奇美拉", "night elf")


def scan_board(text):
    """Which races a blob of caster lines and match context has actually named."""
    blob = (text or "").lower()
    def any_of(markers):
        for m in markers:
            ml = m.lower()
            if ml.isascii():
                if re.search(r"\b" + re.escape(ml) + r"\b", blob):
                    return True
            elif ml in blob:
                return True
        return False
    return {
        "undead": any_of(_UD),
        "orc": any_of(_ORC),
        "far_seer": any_of(_FS),
        "human": any_of(_HU),
        "night_elf": any_of(_NE),
    }


def dog_reading(board):
    """ghouls, wolves, ambiguous, or dog. Wolves only with Orc or a Far Seer and no Undead."""
    undead = bool(board.get("undead"))
    orc = bool(board.get("orc") or board.get("far_seer"))
    if undead and not orc:
        return "ghouls"
    if orc and not undead:
        return "wolves"
    if undead and orc:
        return "ambiguous"
    return "dog"


def match_brief(texts):
    """One note for this line: races seen in the rolling window, and what 狗 means right now."""
    board = scan_board("\n".join(t for t in texts if t))
    races = []
    if board["human"]:
        races.append("Human")
    if board["orc"] or board["far_seer"]:
        races.append("Orc" + (" with a Far Seer" if board["far_seer"] else ""))
    if board["undead"]:
        races.append("Undead")
    if board["night_elf"]:
        races.append("Night Elf")
    seen = ("Races in the match context and recent lines: " + ", ".join(races) + "."
            if races else "No race or hero has been named in the match context or recent lines.")
    reading = dog_reading(board)
    if reading == "ghouls":
        dog = '狗/狗子/小狗 means ghouls. Do not say wolves or spirit wolves.'
    elif reading == "wolves":
        dog = "狗/狗子 means spirit wolves, because Orc or a Far Seer is on screen and Undead is not."
    elif reading == "ambiguous":
        dog = "Both Undead and Orc are on screen. 狗 with the Undead army means ghouls; 狗 with a Far Seer means spirit wolves. Do not default to wolves."
    else:
        dog = '狗/狗子 means "dog". Do not invent ghouls or spirit wolves.'
    return seen + " " + dog


def _prompt_terms(prompt):
    return [t.strip(" .") for t in re.split(r"[,.，、]", prompt) if len(t.strip(" .")) >= 2]


LANGS = {
    "ko": {
        "label": "KO",
        "whisper": "ko",
        "script": HANGUL,
        "google": "ko",
        "mymemory": "ko|en",
        "asr_prompt": KO_ASR_PROMPT,
        "prompt_terms": _prompt_terms(KO_ASR_PROMPT),
        "echo_markers": ["워크래프트3 중계"],
        "junk_sub": KO_JUNK,
        "junk_res": [],
        "system": KO_SYSTEM,
    },
    "zh": {
        "label": "ZH",
        "whisper": "zh",
        "script": HAN,
        "google": "zh-CN",
        "mymemory": "zh-CN|en",
        "asr_prompt": ZH_ASR_PROMPT,
        "prompt_terms": _prompt_terms(ZH_ASR_PROMPT),
        "echo_markers": ["魔兽争霸3解说"],
        "junk_sub": [],
        "junk_res": ZH_JUNK_RES,
        "system": ZH_SYSTEM,
    },
}

SKIP = object()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norm_lang(lang):
    if not lang:
        return "?"
    lang = str(lang).lower().replace("_", "-")
    if lang in ("zh", "zh-cn", "zh-hans", "chinese", "cmn", "mandarin"):
        return "zh"
    if lang in ("ko", "korean"):
        return "ko"
    return lang.split("-")[0]


def is_hallucination(text, lang="ko"):
    """True if the transcript echoes the ASR prompt or is degenerate repetition.

    Korean behavior matches the original checker: the Warcraft prompt header, four or more
    prompt terms with ASCII commas, a repeated word, or a repeated phrase.
    """
    t = (text or "").strip()
    if not t:
        return True
    p = LANGS[lang]
    if any(m in t for m in p["echo_markers"]):
        return True
    hits = sum(1 for term in p["prompt_terms"] if term in t)
    commas = t.count(",") if lang == "ko" else t.count(",") + t.count("，")
    if hits >= 4 and commas >= 3:
        return True
    words = t.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35:
        return True
    if re.search(r"(.{2,15}?)\1{3,}", t.replace(" ", "")):
        return True
    return False


def is_junk(text, lang="ko"):
    p = LANGS[lang]
    if any(j in (text or "") for j in p["junk_sub"]):
        return True
    return any(r.search(text or "") for r in p["junk_res"])


def accept_transcript(detected, text, nsp, src="ko"):
    """Source-language filter: right language, enough script characters, not silence, not a known hallucination."""
    p = LANGS[src]
    if norm_lang(detected) != p["whisper"]:
        return False
    if len(p["script"].findall(text or "")) < 2:
        return False
    if nsp >= 0.6:
        return False
    if is_junk(text, src):
        return False
    return True


def is_skip_translation(text):
    """True when the model declined to translate (empty, or exactly [skip])."""
    if text is None:
        return False
    t = str(text).strip().strip('"').strip("'").strip()
    t = re.sub(r"[。.\s]+$", "", t)
    return t == "" or re.fullmatch(r"\[skip\]", t, re.I) is not None


def clean_spoken_english(text):
    """Drop a trailing self-correction so the voice never says 'wait, no'."""
    if not text:
        return ""
    t = str(text).strip().strip('"')
    t = re.sub(r"(?is)\s*(?:\.\.\.|…)?\s*wait,\s*no\b.*$", "", t)
    t = re.sub(r"(?is)\s*\(\s*wait\b.*?\)\s*$", "", t)
    return t.strip()


def take_live_segment(first, seg_q, max_lag, now=None):
    """Drop segments older than max_lag. If several are waiting, keep only the newest.

    Transcribing the oldest first makes each line wait out the whole backlog, so the
    lag ratchets up on a busy cast. The stale ones are discarded before ASR.
    """
    now = time.time() if now is None else now
    batch = [first]
    while True:
        try:
            batch.append(seg_q.get_nowait())
        except queue.Empty:
            break
    live = [s for s in batch if now - s[0] <= max_lag]
    dropped = len(batch) - len(live)
    if not live:
        return None, dropped
    if len(live) > 1:
        dropped += len(live) - 1
        return max(live, key=lambda s: s[0]), dropped
    return live[0], dropped


class Voice:
    """Speak one line. After repeated Edge failures, stop calling Edge and use the fallback."""

    def __init__(self, edge=None, eleven=None, say=None, fail_limit=2):
        self.edge, self.eleven, self.say = edge, eleven, say
        self.fail_limit = fail_limit
        self.edge_fails = 0
        self.edge_off = False
        self._announced = False

    def _disable_edge(self):
        if self.edge_off:
            return
        self.edge_off = True
        if self._announced:
            return
        self._announced = True
        where = "ElevenLabs Flash via OpenRouter" if self.eleven else ("macOS say" if self.say else "no further voice")
        log(f"Edge TTS disabled after {self.edge_fails} handshake failures; using {where}")

    def __call__(self, text, rate="+0%"):
        if not self.edge_off and self.edge is not None:
            try:
                pcm = self.edge(text, rate)
                self.edge_fails = 0
                return pcm
            except Exception as e:
                self.edge_fails += 1
                log("edge-tts failed:", type(e).__name__)
                if self.edge_fails >= self.fail_limit:
                    self._disable_edge()
        if self.eleven is not None:
            try:
                return self.eleven(text)
            except Exception as e:
                log("openrouter tts failed:", type(e).__name__)
        if self.say is not None:
            try:
                return self.say(text)
            except Exception as e:
                log("say failed:", type(e).__name__)
        return None


def time_of(marks, idx):
    """Capture-clock time of sample index idx. marks are (start_sample, t_start), ascending."""
    if not marks:
        return None
    t = None
    for start, t_start in marks:
        if start > idx:
            break
        t = t_start + (idx - start) / SR
    if t is None:
        start, t_start = marks[0]
        t = t_start + (idx - start) / SR
    return t


class SampleClock:
    """Maps captured samples onto time.monotonic().

    A decode burst stays contiguous and slightly ahead of the wall clock, so a delay
    line can release it at realtime. A stall shifts the origin forward so the next
    audio is not dumped as one overdue lump.
    """

    def __init__(self, now=None):
        self._now = now or time.monotonic
        self.t0 = None
        self.n = 0

    def stamp(self, n):
        if n <= 0:
            return self._now() if self.t0 is None else self.t0 + self.n / SR
        now = self._now()
        if self.t0 is None:
            self.t0 = now - n / SR
        else:
            predicted_end = self.t0 + (self.n + n) / SR
            if predicted_end < now - 0.3:
                self.t0 += now - predicted_end
        self.n += n
        return self.t0 + self.n / SR


# ---------------------------------------------------------------- audio input
class _FD:
    def __init__(self, fd): self.fd = fd
    def read(self, n=-1): return self.fd.read(n if n and n > 0 else 65536)
    def readable(self): return True


class _Prefixed:
    """Replay bytes already consumed by container sniffing."""

    def __init__(self, head, fd):
        self._head = head or b""
        self._fd = fd

    def read(self, n=-1):
        if n is None or n < 0:
            n = 65536
        out = b""
        if self._head:
            take = self._head[:n]
            self._head = self._head[n:]
            out = take
            if len(out) >= n:
                return out
            n -= len(out)
        chunk = self._fd.read(n)
        return out + (chunk or b"")

    def readable(self):
        return True


def sniff_container(fd):
    """Douyu, Huya and Douyin are FLV. SOOP/Afreeca via streamlink is MPEG-TS."""
    head = b""
    while len(head) < 4:
        b = fd.read(4 - len(head))
        if not b:
            break
        head += b
    if head.startswith(b"FLV"):
        fmt = "flv"
    elif head.startswith(b"OggS"):
        fmt = "ogg"
    elif head.startswith(b"RIFF"):
        fmt = "wav"
    elif head[:1] == b"\x47":
        fmt = "mpegts"
    else:
        fmt = "mpegts"
        if head:
            log(f"unrecognized stream header {head[:8]!r}; trying mpegts")
    return fmt, _Prefixed(head, fd)


def _remember_stream(meta, plugin, plugin_name):
    title = author = ""
    try:
        title = plugin.get_title() or ""
        author = plugin.get_author() or ""
    except Exception:
        pass
    if meta is not None:
        meta["title"] = title
        meta["author"] = author
        meta["plugin"] = plugin_name
    bits = [b for b in (plugin_name, author, title) if b]
    if bits:
        log("stream info:", " | ".join(bits))


def audio_source(url, quality, out_q, stop, tap_q=None, meta=None):
    """Pull the live stream with streamlink, decode audio to 16 kHz mono float32.

    Each chunk is (capture-clock end time, pcm). tap_q, when set, gets the same
    pairs for the overdub delay line.
    """
    import streamlink
    clock = SampleClock()
    while not stop.is_set():
        try:
            session = streamlink.Streamlink()
            session.set_option("http-timeout", 20)
            plugin_name, plugin_cls, resolved = session.resolve_url(url)
            plugin = plugin_cls(session, resolved)
            streams = plugin.streams()
            if not streams:
                log("stream offline / no streams, retrying in 15s")
                time.sleep(15)
                continue
            _remember_stream(meta, plugin, plugin_name)
            if quality in streams:
                picked = quality
            elif "worst" in streams:
                picked = "worst"
            else:
                picked = next(iter(streams))
            raw = streams[picked].open()
            fmt, replay = sniff_container(_FD(raw))
            log(f"connected ({picked}, {fmt})")
            kwargs = {"format": fmt}
            if fmt == "flv":
                kwargs["options"] = {"fflags": "nobuffer", "flags": "low_delay"}
            c = av.open(replay, **kwargs)
            rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
            for frame in c.decode(audio=0):
                if stop.is_set():
                    break
                for f in rs.resample(frame):
                    pcm = f.to_ndarray().reshape(-1).astype(np.float32)
                    if len(pcm) == 0:
                        continue
                    t_end = clock.stamp(len(pcm))
                    out_q.put((t_end, pcm))
                    if tap_q is not None:
                        tap_q.put((t_end, pcm))
        except Exception as e:
            log("input error:", repr(e), "- reconnecting in 3s")
            time.sleep(3)


# ---------------------------------------------------------------- VAD segmenter
def segmenter(in_q, seg_q, stop, max_len=12.0, min_len=0.6):
    from silero_vad import load_silero_vad, VADIterator
    import torch
    model = load_silero_vad()
    vad = VADIterator(model, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=450, speech_pad_ms=120)
    buf = np.zeros(0, np.float32)
    cur = None  # list of chunks while in speech
    pre = []    # small pre-roll
    samples_in = 0
    processed = 0
    marks = []  # (start_sample, t_start) on the capture clock
    while not stop.is_set():
        try:
            t_end, chunk = in_q.get(timeout=1)
        except queue.Empty:
            continue
        if len(chunk) == 0:
            continue
        marks.append((samples_in, t_end - len(chunk) / SR))
        samples_in += len(chunk)
        while len(marks) > 2 and marks[1][0] < samples_in - int(SR * 30):
            marks.pop(0)
        buf = np.concatenate([buf, chunk])
        while len(buf) >= 512:
            ch, buf = buf[:512], buf[512:]
            processed += 512
            ev = vad(torch.from_numpy(ch), return_seconds=False)
            if ev and "start" in ev and cur is None:
                cur = pre[-4:] + [ch]
            elif cur is not None:
                cur.append(ch)
                if (ev and "end" in ev) or len(cur) * 512 / SR >= max_len:
                    seg = np.concatenate(cur)
                    if len(seg) / SR >= min_len:
                        seg_q.put((time.time(), seg, time_of(marks, processed - len(seg))))
                    cur = None
                    if not (ev and "end" in ev):
                        vad.reset_states()
            pre.append(ch)
            pre = pre[-4:]


# ---------------------------------------------------------------- ASR
class ASR:
    def __init__(self, model, prompt=True, ctx_lines=0, lang="ko"):
        self.mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.prompt, self.ctx_lines, self.prev = prompt, ctx_lines, []
        self.lang = lang if lang in LANGS else "ko"
        self.profile = LANGS[self.lang]
        self.code = self.profile["whisper"]
        if self.mlx:
            import mlx_whisper
            self.m = mlx_whisper
            # large-v3 cannot keep up with back-to-back Mandarin casting. Turbo is the zh default.
            if model:
                self.name = model
            elif self.lang == "zh":
                self.name = "mlx-community/whisper-large-v3-turbo"
            else:
                self.name = "mlx-community/whisper-large-v3-mlx"
            self.m.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=self.name, language=self.code)  # warm-up / download
        else:
            from faster_whisper import WhisperModel
            self.name = model or "small"
            self.m = WhisperModel(self.name, device="cpu", compute_type="int8")

    def _initial_prompt(self):
        if not self.prompt:
            return None
        p = self.profile["asr_prompt"]
        if self.ctx_lines and self.prev:
            p += " " + " ".join(self.prev[-self.ctx_lines:])
        return p

    def transcribe(self, audio):
        """returns (lang, text, no_speech_prob)"""
        ip = self._initial_prompt()
        if self.mlx:
            r = self.m.transcribe(audio, path_or_hf_repo=self.name, condition_on_previous_text=False,
                                  temperature=0.0, verbose=None, initial_prompt=ip)
            segs = r.get("segments", [])
            nsp = min([s.get("no_speech_prob", 0) for s in segs] or [1])
            if any(s.get("compression_ratio", 0) > 2.4 for s in segs):
                nsp = 1
            lang, text = norm_lang(r.get("language")), r.get("text", "").strip()
        else:
            segs, info = self.m.transcribe(audio, beam_size=1, condition_on_previous_text=False, vad_filter=False,
                                           initial_prompt=ip)
            segs = list(segs)
            nsp = min([s.no_speech_prob for s in segs] or [1])
            lang = norm_lang(info.language if info.language_probability >= 0.5 else "?")
            text = "".join(s.text for s in segs).strip()
        if is_hallucination(text, self.lang):
            return lang, "", 1.0
        if lang == self.code and text:
            self.prev = (self.prev + [text])[-2:]
        return lang, text, nsp

    def translate(self, audio):
        if self.mlx:
            return self.m.transcribe(audio, path_or_hf_repo=self.name, language=self.code, task="translate",
                                     temperature=0.0, verbose=None)["text"].strip()
        segs, _ = self.m.transcribe(audio, language=self.code, task="translate", beam_size=1)
        return "".join(s.text for s in segs).strip()


# ---------------------------------------------------------------- translation
def google_url(text, sl="ko"):
    import urllib.parse
    return ("https://translate.googleapis.com/translate_a/single?client=gtx&sl="
            + sl + "&tl=en&dt=t&q=" + urllib.parse.quote(text))


def mymemory_url(text, pair="ko|en"):
    import urllib.parse
    return "https://api.mymemory.translated.net/get?langpair=" + pair + "&q=" + urllib.parse.quote(text)


def google_translate(text, sl="ko"):
    import json, urllib.request
    req = urllib.request.Request(google_url(text, sl), headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=5))
    return "".join(x[0] for x in data[0] if x and x[0]).strip()


def mymemory_translate(text, pair="ko|en"):
    import json, urllib.request
    d = json.load(urllib.request.urlopen(mymemory_url(text, pair), timeout=5))
    return (d.get("responseData") or {}).get("translatedText", "").strip()


def translate_text(text, sl="ko", pair="ko|en", script=None):
    """Google (with one retry) -> MyMemory. Returns None if all fail."""
    script = HANGUL if script is None else script
    for fn, tries, arg in ((google_translate, 2, sl), (mymemory_translate, 1, pair)):
        for _ in range(tries):
            try:
                en = fn(text, arg)
                if en and not script.search(en):
                    return en
            except Exception as e:
                log(f"  ({fn.__name__} {type(e).__name__})")
            time.sleep(0.4)
    return None


# ---------------------------------------------------------------- LLM translation (OpenRouter)
class LLMTranslator:
    def __init__(self, key, model, ctx=5, system=None, extra_context=""):
        self.key, self.model, self.ctx = key, model, ctx
        base = KO_SYSTEM if system is None else system
        self.system = str(base)  # copy; later stream-title notes must not mutate LANGS
        self.extra_context = (extra_context or "").strip()
        if self.extra_context:
            self.system += ("\n\nMatch context from the listener (who is on stream, the series, the map): "
                            + self.extra_context)
        self.history = []  # (source, en)
        self.recent = deque(maxlen=8)  # source lines, including ones the model skipped
        self.stream_title = ""
        self.stream_author = ""
        self.tokens_in = self.tokens_out = 0
        self.cost = 0.0
        self._stream_note = False

    def add_stream_context(self, title, author):
        self.stream_title = title or self.stream_title
        self.stream_author = author or self.stream_author
        if self._stream_note or not (title or author):
            return
        lines = []
        if title:
            lines.append("Stream title: " + title)
        if author:
            lines.append("Streamer: " + author)
        self.system += "\n" + "\n".join(lines)
        self._stream_note = True

    def note_for(self, text):
        return match_brief([self.extra_context, self.stream_title, self.stream_author,
                            *self.recent, text])

    def messages_for(self, text):
        system = self.system + "\n\n" + self.note_for(text)
        msgs = [{"role": "system", "content": system}]
        for src, en in self.history[-self.ctx:]:
            msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": en}]
        msgs.append({"role": "user", "content": text})
        return msgs

    def __call__(self, text):
        import json, urllib.request
        self.recent.append(text)
        msgs = self.messages_for(text)
        body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": 300, "temperature": 0.2,
                           "reasoning": {"enabled": False},
                           "provider": {"sort": "latency"},
                           "usage": {"include": True}}).encode()
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
            "X-Title": "kodub"})
        d = json.load(urllib.request.urlopen(req, timeout=6))
        en = (d["choices"][0]["message"].get("content") or "").strip().strip('"')
        u = d.get("usage") or {}
        self.tokens_in += u.get("prompt_tokens", 0)
        self.tokens_out += u.get("completion_tokens", 0)
        self.cost += float(u.get("cost") or 0)
        if is_skip_translation(en):
            return SKIP
        if en:
            self.history.append((text, en))
        return en


# ---------------------------------------------------------------- TTS + playback
def eleven_pcm(text, voice_id, key):
    """ElevenLabs low-latency TTS (Flash v2.5), raw 24 kHz PCM."""
    import json, urllib.request
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000&optimize_streaming_latency=3"
    body = json.dumps({"text": text, "model_id": "eleven_flash_v2_5"}).encode()
    req = urllib.request.Request(url, data=body, headers={"xi-api-key": key, "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=15).read()
    return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768


def openrouter_tts_pcm(text, model, voice, key):
    """TTS via OpenRouter /audio/speech (e.g. elevenlabs/eleven-flash-v2.5). Returns 24 kHz mono float32."""
    import json, urllib.request
    body = json.dumps({"model": model, "input": text, "voice": voice, "response_format": "mp3"}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/audio/speech", data=body, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "X-Title": "kodub"})
    mp3 = urllib.request.urlopen(req, timeout=10).read()
    c = av.open(io.BytesIO(mp3))
    rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
    if not out:
        raise RuntimeError("no audio")
    return np.concatenate(out)


def say_pcm(text):
    """macOS built-in TTS fallback."""
    import os, subprocess, tempfile
    f = tempfile.mktemp(suffix=".aiff")
    subprocess.run(["say", "-o", f, text], check=True)
    try:
        c = av.open(f)
        rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
        return np.concatenate([x.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for x in rs.resample(fr)])
    finally:
        os.unlink(f)


def tts_pcm(text, voice, rate):
    import edge_tts
    async def go():
        b = bytearray()
        async for ch in edge_tts.Communicate(text, voice, rate=rate).stream():
            if ch["type"] == "audio":
                b.extend(ch["data"])
        return bytes(b)
    mp3 = asyncio.run(go())
    c = av.open(io.BytesIO(mp3))
    rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


class Speaker:
    """One persistent output stream at the device's native rate with a generous buffer.

    The old code called sd.play(pcm, 24000) per line: a fresh tiny-buffer (~20 ms) stream at 24 kHz on a
    44.1/48 kHz (often AirPlay/Bluetooth) device, so CoreAudio resampled on the fly and the Python callback
    had to keep up while Whisper/VAD hogged the CPU/GPU -> crackles. Here we resample once with libswresample,
    keep one stream open, use large blocks + ~0.4 s latency, add short fades and a soft limiter."""

    def __init__(self, volume=1.0, device=None, latency=0.4, blocksize=2048):
        import sounddevice as sd
        self.sd, self.volume = sd, volume
        info = sd.query_devices(device, "output")
        self.sr = int(info["default_samplerate"]) or 48000
        self.buf = np.zeros(0, np.float32)
        self.lock = threading.Lock()
        self.idle = threading.Event(); self.idle.set()
        self.underflows = self.callbacks = self.late = 0
        self._last = None
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32", device=device,
                                      blocksize=blocksize, latency=latency, callback=self._cb)
        self.stream.start()
        self.name = info["name"]
        self.slack = self.stream.latency + blocksize / self.sr  # max callback gap before audio runs dry

    def _cb(self, out, frames, t, status):
        self.callbacks += 1
        now = time.perf_counter()
        if status.output_underflow or (self._last and now - self._last > self.slack):
            self.underflows += 1
        self._last = now
        with self.lock:
            n = min(frames, len(self.buf))
            out[:n, 0] = self.buf[:n]
            out[n:, 0] = 0
            self.buf = self.buf[n:]
            if not len(self.buf):
                self.idle.set()

    def resample(self, pcm, rate=24000):
        if rate == self.sr or not len(pcm):
            return pcm.astype(np.float32)
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm, np.float32).reshape(1, -1), format="flt", layout="mono")
        fr.sample_rate = rate
        rs = av.AudioResampler(format="flt", layout="mono", rate=self.sr)
        out = [f.to_ndarray().reshape(-1) for f in rs.resample(fr)] + [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

    def prepare(self, pcm, rate=24000):
        x = self.resample(np.nan_to_num(pcm), rate) * self.volume
        f = min(int(self.sr * 0.008), len(x) // 2)  # 8 ms fade in/out: no click at line edges
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            x[:f] *= ramp
            x[-f:] *= ramp[::-1]
        return np.tanh(x * 1.0).astype(np.float32) if np.abs(x).max() > 0.95 else x  # soft limit, never hard-clip

    def play(self, pcm, rate=24000):
        x = self.prepare(pcm, rate)
        with self.lock:
            self.buf = np.concatenate([self.buf, x, np.zeros(int(self.sr * 0.15), np.float32)])
            self.idle.clear()
        self.idle.wait()

    def close(self):
        self.idle.wait(5)
        time.sleep(0.5)
        self.stream.stop()
        self.stream.close()


SPEAKER = None


def player(play_q, stop, mute, volume, save, device=None, latency=0.4):
    sp = None
    if not mute:
        global SPEAKER
        sp = SPEAKER = Speaker(volume, device, latency)
        log(f"audio out: {sp.name} @ {sp.sr} Hz, latency {sp.stream.latency:.2f}s")
    saved = []
    while not stop.is_set():
        try:
            pcm = play_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if save is not None:
            saved.append(pcm)
            saved.append(np.zeros(6000, np.float32))
        if sp:
            before = sp.underflows
            sp.play(pcm)
            if sp.underflows > before:
                log(f"  (audio underflow x{sp.underflows - before})")
    if save is not None and saved:
        _write_wav(save, saved)


def _write_wav(path, chunks):
    import wave
    a = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(a.tobytes())


class DelayLine:
    """Holds 16 kHz chunks stamped with a capture-clock end time and releases them `delay` seconds later."""

    def __init__(self, delay):
        self.delay = delay
        self.chunks = deque()

    def push(self, t_end, pcm):
        if len(pcm) == 0:
            return
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        self.chunks.append([t_end - len(pcm) / SR, pcm, 0])

    def pull_due(self, now):
        cutoff = now - self.delay
        while self.chunks:
            t_start, pcm, _off = self.chunks[0]
            if t_start + len(pcm) / SR < cutoff - 1.0:
                self.chunks.popleft()  # stale audio after a stall; don't fast-forward it all
                continue
            break
        parts = []
        while self.chunks:
            t_start, pcm, off = self.chunks[0]
            due = int((cutoff - t_start) * SR) - off
            if due <= 0:
                break
            due = min(due, len(pcm) - off)
            parts.append(pcm[off:off + due])
            if off + due >= len(pcm):
                self.chunks.popleft()
            else:
                self.chunks[0][2] = off + due
                break
        if not parts:
            return np.zeros(0, np.float32)
        return np.concatenate(parts)


def mix_block(dev, start_at, sr, dubs, duck):
    """Mix device-rate stream audio with scheduled English dubs. Duck the stream under each dub."""
    dev = np.nan_to_num(np.ascontiguousarray(dev, dtype=np.float32))
    if len(dev) == 0:
        return dev
    add = np.zeros(len(dev), np.float32)
    gain = np.ones(len(dev), np.float32)
    end_at = start_at + len(dev) / float(sr)
    for d in dubs:
        pcm = d["pcm"]
        cur = d["cur"]
        if cur >= len(pcm):
            continue
        sample_at = d["at"] + cur / float(sr)
        if sample_at >= end_at:
            continue
        offset = 0 if sample_at <= start_at else int(round((sample_at - start_at) * sr))
        if offset >= len(dev):
            continue
        n = min(len(pcm) - cur, len(dev) - offset)
        if n <= 0:
            continue
        sl = slice(offset, offset + n)
        add[sl] += pcm[cur:cur + n]
        gain[sl] = duck
        d["cur"] = cur + n
    out = dev * gain + add
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1:
        out = (out / peak).astype(np.float32)
    return out


def overdub_loop(tap_q, dub_q, stop, volume, device, latency, delay, duck, save):
    """Play stream audio `delay` seconds late, with the English dub mixed on top and the stream ducked."""
    global SPEAKER
    sp = SPEAKER = Speaker(volume, device, latency)
    log(f"audio out: {sp.name} @ {sp.sr} Hz, latency {sp.stream.latency:.2f}s")
    log(f"overdub: mute the browser. Stream audio is held {delay:.1f}s "
        f"(about {delay + sp.stream.latency:.1f}s behind the live picture, including the output buffer) "
        f"and ducked to {duck:.0%} while the English voice speaks.")
    line = DelayLine(delay)
    dubs = []
    saved = []
    while not stop.is_set():
        while True:
            try:
                t_end, pcm = tap_q.get_nowait()
            except queue.Empty:
                break
            line.push(t_end, pcm)
        while True:
            try:
                play_at, pcm24 = dub_q.get_nowait()
            except queue.Empty:
                break
            dubs.append({"at": play_at, "pcm": sp.prepare(pcm24, 24000), "cur": 0})
            if save is not None:
                saved.append(np.asarray(pcm24, np.float32))
                saved.append(np.zeros(6000, np.float32))
            late = time.monotonic() - play_at
            if late > 0.35:
                log(f"  (overdub {late:.1f}s late — raise --overdub-delay if the English trails the caster)")
        due = line.pull_due(time.monotonic())
        if len(due):
            dev = sp.resample(due, SR) * sp.volume
            with sp.lock:
                buffered = len(sp.buf)
            start_at = time.monotonic() + buffered / float(sp.sr)
            mixed = mix_block(dev, start_at, sp.sr, dubs, duck)
            dubs[:] = [d for d in dubs if d["cur"] < len(d["pcm"])]
            with sp.lock:
                sp.buf = np.concatenate([sp.buf, mixed])
                if len(sp.buf):
                    sp.idle.clear()
        else:
            time.sleep(0.02)
    if save is not None and saved:
        _write_wav(save, saved)


def speak_loop(jobs, stop, voice, play_q, dub_q, stats):
    """Synthesize while the main thread is already on the next line's ASR."""
    while not stop.is_set():
        try:
            job = jobs.get(timeout=0.3)
        except queue.Empty:
            continue
        if job is None:
            break
        age = time.time() - job["t_seg"]
        if age > job["max_lag"]:
            log(f"[skip, {age:.0f}s behind]")
            continue
        t0 = time.time()
        pcm = voice(job["en"], job["rate"])
        if pcm is None or len(pcm) == 0:
            log(f"  - no audio for: {job['en'][:40]}")
            continue
        stats["tts"].append(time.time() - t0)
        stats["chars"][0] += len(job["en"])
        d = time.time() - job["t_seg"] - job["cmp_t"]
        stats["lat"].append(d)
        if job["dub"] is not None:
            play_at = (job["t_start"] + job["overdub_delay"]) if job["t_start"] is not None else time.monotonic()
            job["dub"].put((play_at, pcm))
        else:
            play_q.put(pcm)
        log(f"{job['label']}: {job['text']}\n         EN: {job['en']}   (ready {d:.1f}s after speech ended)")


# ---------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description="Live Korean or Mandarin -> English voice dub for a stream URL")
    ap.add_argument("url")
    ap.add_argument("--lang", choices=["ko", "zh"], default="ko",
                    help="source language: ko (Korean, default) or zh (Mandarin)")
    ap.add_argument("--context", default="",
                    help='extra match context for the translator, e.g. "Fly100%% vs Lyn, Bo3"')
    ap.add_argument("--quality", default="worst",
                    help="stream quality (default worst). Douyu names look like 900k; worst is the lowest bitrate")
    ap.add_argument("--asr-model", "--model", dest="model", default=None,
                    help="whisper model (Apple Silicon default: large-v3 for Korean, large-v3-turbo for Mandarin; 'small' elsewhere)")
    ap.add_argument("--no-asr-prompt", action="store_true", help="disable the Warcraft 3 vocabulary prompt for Whisper")
    ap.add_argument("--asr-context", type=int, default=0,
                    help="also feed the last N (0-2) source-language lines to Whisper as context")
    ap.add_argument("--voice", default="en-US-GuyNeural", help="edge-tts voice, e.g. en-US-AriaNeural")
    ap.add_argument("--eleven-voice", default="JBFqnCBsd6RMkjVDRZzb",
                    help="ElevenLabs voice id (George; used only when ELEVENLABS_API_KEY is set)")
    ap.add_argument("--tts", choices=["edge", "eleven"], default="edge",
                    help="eleven = ElevenLabs Flash v2.5 via OpenRouter (voice george). Best match for --overdub")
    ap.add_argument("--tts-model", default="elevenlabs/eleven-flash-v2.5", help="OpenRouter speech model for --tts eleven")
    ap.add_argument("--tts-voice", default="george", help="voice name for --tts eleven (a consistent caster-style voice)")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--out-device", default=None, help="output device name/index (default: system output)")
    ap.add_argument("--out-latency", type=float, default=0.4, help="output buffer in seconds (raise if it crackles)")
    ap.add_argument("--overdub", action="store_true",
                    help="play the stream audio yourself, delayed, ducked under the English dub. Mute the browser.")
    ap.add_argument("--overdub-delay", type=float, default=7.0,
                    help="seconds to hold stream audio in --overdub so the dub can land on the caster's line")
    ap.add_argument("--duck", type=float, default=0.22,
                    help="stream gain while the English voice is speaking, 0-1 (game audio ducks too; it is one mix)")
    ap.add_argument("--translator", choices=["google", "whisper"], default="google",
                    help="google = transcribe then translate (better); whisper = whisper's built-in translate")
    ap.add_argument("--llm-model", default="anthropic/claude-haiku-5.5", help="OpenRouter model slug")
    ap.add_argument("--compare", action="store_true", help="also show Google's translation for each line (testing)")
    ap.add_argument("--max-lag", type=float, default=8.0,
                    help="before ASR, drop speech older than this many seconds and keep only the newest line if several are waiting")
    ap.add_argument("--mute", action="store_true", help="don't play audio, just print (testing)")
    ap.add_argument("--save", default=None, help="also write dubbed audio to this .wav")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (testing)")
    return ap


def main():
    a = build_parser().parse_args()
    prof = LANGS[a.lang]
    a.overdub_delay = max(0.0, a.overdub_delay)
    a.duck = min(1.0, max(0.0, a.duck))
    if a.overdub and a.mute:
        log("--mute set; ignoring --overdub")
        a.overdub = False

    log("loading models (first run downloads ~1.5 GB)...")
    asr = ASR(a.model, prompt=not a.no_asr_prompt, ctx_lines=min(a.asr_context, 2), lang=a.lang)
    log("ASR ready:", asr.name, "(MLX)" if asr.mlx else "(CPU)", "lang", a.lang)
    gt = a.translator == "google"
    okey = load_openrouter_key()
    llm = LLMTranslator(okey, a.llm_model, system=prof["system"], extra_context=a.context) if okey else None
    log("translator:", f"{a.llm_model} via OpenRouter (Google fallback)" if llm else "Google (no OpenRouter key found)")
    if a.context:
        log("context:", a.context)
        if not llm:
            log("no OpenRouter key; --context is only used by the LLM translator")
    llm_t = []
    import os
    ekey = os.environ.get("ELEVENLABS_API_KEY")
    say = say_pcm if platform.system() == "Darwin" else None

    def eleven(text, _rate=None):
        err = None
        if okey:
            try:
                return openrouter_tts_pcm(text, a.tts_model, a.tts_voice, okey)
            except Exception as e:
                err = e
                log("openrouter tts failed:", type(e).__name__)
        if ekey:
            return eleven_pcm(text, a.eleven_voice, ekey)
        if err:
            raise err
        raise RuntimeError("no eleven key")

    have_eleven = bool(okey or ekey)
    if a.tts == "eleven" and have_eleven:
        voice = Voice(edge=None, eleven=eleven, say=say)
        log("TTS:", a.tts_model if okey else "ElevenLabs direct", "voice", a.tts_voice if okey else a.eleven_voice)
    else:
        if a.tts == "eleven":
            log("no ElevenLabs key; using Edge")
        voice = Voice(edge=lambda text, rate: tts_pcm(text, a.voice, rate),
                      eleven=eleven if have_eleven else None, say=say)
        log("TTS: Edge", a.voice, "(edge-tts>=7.3.1); fallback",
            "ElevenLabs" if have_eleven else ("macOS say" if say else "none"))
    if a.overdub and a.tts != "eleven":
        log("overdub tip: --tts eleven uses ElevenLabs Flash v2.5 (voice george, same OpenRouter key) "
            "and usually finishes inside the hold")
    stats = {"tts": [], "lat": [], "chars": [0]}

    stop = threading.Event()
    in_q, seg_q, play_q = queue.Queue(), queue.Queue(), queue.Queue()
    meta = {}
    tap_q = dub_q = None
    if a.overdub:
        tap_q, dub_q = queue.Queue(), queue.Queue()
    threading.Thread(target=audio_source, args=(a.url, a.quality, in_q, stop, tap_q, meta), daemon=True).start()
    threading.Thread(target=segmenter, args=(in_q, seg_q, stop), daemon=True).start()
    if a.overdub:
        threading.Thread(target=overdub_loop, args=(
            tap_q, dub_q, stop, a.volume, a.out_device, a.out_latency, a.overdub_delay, a.duck, a.save,
        ), daemon=True).start()
    else:
        threading.Thread(target=player, args=(
            play_q, stop, a.mute, a.volume, a.save, a.out_device, a.out_latency,
        ), daemon=True).start()
    jobs = queue.Queue()
    speaker = threading.Thread(target=speak_loop, args=(jobs, stop, voice, play_q, dub_q, stats), daemon=True)
    speaker.start()

    t_end = time.time() + a.duration if a.duration else None
    try:
        while not (t_end and time.time() > t_end):
            try:
                got = seg_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if llm:
                llm.add_stream_context(meta.get("title") or "", meta.get("author") or "")
            got, dropped = take_live_segment(got, seg_q, a.max_lag)
            if dropped:
                log(f"[skip {dropped} behind]")
            if got is None:
                continue
            t_seg, seg, t_start = got
            lang, text, nsp = asr.transcribe(seg)
            if not accept_transcript(lang, text, nsp, a.lang):
                if text:
                    log(f"  - skip [{lang}] {text[:70]}")
                continue
            en = None
            cmp_t = 0.0
            if gt:
                if llm:
                    t0 = time.time()
                    try:
                        en = llm(text)
                        llm_t.append(time.time() - t0)
                    except Exception as e:
                        log(f"  (LLM {type(e).__name__}, falling back to Google)")
                        en = None
                    if en is SKIP:
                        log(f"  - llm skip [{text[:60]}]")
                        continue
                    if prof["script"].search(en or ""):
                        en = None
                if a.compare and en:
                    t0 = time.time()
                    try:
                        g = google_translate(text, prof["google"])
                    except Exception as e:
                        g = f"<{type(e).__name__}>"
                    cmp_t = time.time() - t0
                    log(f"  google [{cmp_t:.2f}s]: {g}")
                if not en:
                    en = translate_text(text, sl=prof["google"], pair=prof["mymemory"], script=prof["script"])
            if not en:
                en = asr.translate(seg)
            en = clean_spoken_english(en)
            if not en or en is SKIP or prof["script"].search(en) or not re.search(r"[A-Za-z]", en):
                log(f"  - translation failed for: {text[:60]}")
                continue
            backlog = jobs.qsize() + seg_q.qsize()
            rate = f"+{min(10 + 15 * backlog, 50)}%"
            jobs.put({
                "t_seg": t_seg, "t_start": t_start, "text": text, "en": en, "cmp_t": cmp_t,
                "rate": rate, "label": prof["label"], "max_lag": a.max_lag,
                "dub": dub_q, "overdub_delay": a.overdub_delay,
            })
    except KeyboardInterrupt:
        pass
    jobs.put(None)
    speaker.join(8)
    stop.set()
    tts_t = stats["tts"]
    lat = stats["lat"]
    tts_chars = stats["chars"][0]
    if llm_t:
        log(f"LLM call time: median {np.median(llm_t):.2f}s, max {max(llm_t):.2f}s; "
            f"tokens in/out {llm.tokens_in}/{llm.tokens_out}, cost ${llm.cost:.5f} for {len(llm_t)} lines")
    if tts_t:
        log(f"TTS time: median {np.median(tts_t):.2f}s, max {max(tts_t):.2f}s, "
            f"{tts_chars} chars spoken")
    if lat:
        log(f"latency (speech end -> English audio ready): median {np.median(lat):.1f}s, max {max(lat):.1f}s, n={len(lat)}")
    if a.overdub and SPEAKER is not None:
        total = a.overdub_delay + SPEAKER.stream.latency
        log(f"overdub lag vs the live picture: about {total:.1f}s "
            f"({a.overdub_delay:.1f}s hold + {SPEAKER.stream.latency:.2f}s output buffer)")
    elif lat:
        log("normal mode adds no extra hold; that figure is how long after the caster finishes a line the English is ready")
    if SPEAKER:
        log(f"audio: {SPEAKER.underflows} underflows/dropouts in {SPEAKER.callbacks} output callbacks")
    time.sleep(1.5)


def load_openrouter_key():
    import os
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for path in ("~/kodub/.env", "~/.env", "~/.hermes/.env"):
        try:
            for line in open(os.path.expanduser(path)):
                line = line.strip()
                if line.startswith(("OPENROUTER_API_KEY=", "export OPENROUTER_API_KEY=")):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


if __name__ == "__main__":
    main()

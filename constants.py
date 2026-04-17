"""

Constants used across the project

Usage:
    from constants import XYZ

"""

# First Congress for which amendment data is available in the raw dataset
AMENDMENTS_START_CONGRESS: int = 108

# Keywords

CHINA_KEYWORDS = [
    "china", "chinese", "prc", "people's republic of china", "beijing", "shanghai", "shenzhen",
    "guangzhou", "national people's congress", "npc", "politburo", "chinese communist party", "ccp",
    "communist party of china", "state council of china", "pboc", "people's bank of china", "mofcom",
    "ministry of commerce china", "ministry of foreign affairs china", "mfa china", "mss",
    "ministry of state security", "pla", "people's liberation army", "pla navy", "pla air force", "rocket force",
    "xi jinping", "li keqiang", "li qiang", "hu jintao", "xinhua", "people's daily", "cgtn", "hong kong", "taiwan",
    "indo-pacific", "dalai lama", "tibet", "xinjiang", "uyghur", "uighur", "human rights", "taipei", "taiwanese",
    "panama canal", "tariff", "rare earth", "belt and road", "bri", "third neighbor","foreign adversary", "foreign entity"
    "foreign adversaries", "foreign entities", "foreign adversary's", "foreign entity's"
]
# ------------------------------------------------------------
# CRIME NEWS ANALYSIS PIPELINE
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import re
import regex as re2
from collections import Counter
from typing import List, Dict, Any

# ------------------------------------------------------------
# 1. Crime / Justice Term Definitions
# ------------------------------------------------------------

crime_justice_terms = {
    "officials": {
        "terms": [
            "district attorney", "da's office", "prosecutor", "public defender",
            "judge", "police chief", "sheriff", "brooke jenkins", "chesa boudin",
            "pamela price", "wagstaffe", "nancy o'malley"
        ],
        "weight": 5
    },
    "violent_crime": {
        "terms": [
            "murder", "homicide", "assault", "shooting", "stabbing", "armed robbery",
            "violent crime", "domestic violence", "sexual assault", "rape",
            "kidnapping", "human trafficking"
        ],
        "weight": 4
    },
    "drug_crime": {
        "terms": [
            "fentanyl", "narcotics", "drug trafficking", "drug dealing",
            "opioid", "overdose", "narcan", "controlled substance",
            "drug possession", "drug sales", "drug bust"
        ],
        "weight": 3
    },
    "property_crime": {
        "terms": [
            "robbery", "burglary", "theft", "shoplifting", "auto theft",
            "vandalism", "property crime", "stolen property", "breaking and entering",
            "smash and grab", "retail theft", "grand theft"
        ],
        "weight": 3
    },
    "legal_process": {
        "terms": [
            "arrest", "investigation", "trial", "court hearing", "sentencing",
            "conviction", "plea deal", "indictment", "warrant", "bail",
            "custody", "criminal charges", "felony", "misdemeanor"
        ],
        "weight": 3
    },
    "law_enforcement": {
        "terms": [
            "police", "law enforcement", "officer", "patrol", "precinct",
            "task force", "detective", "investigation unit", "swat",
            "police department", "pd", "sheriff's department"
        ],
        "weight": 2
    },
    "justice_system": {
        "terms": [
            "criminal justice", "courthouse", "jail", "prison", "probation",
            "parole", "rehabilitation", "recidivism", "inmate", "detention",
            "corrections", "criminal record"
        ],
        "weight": 2
    },
    "public_safety": {
        "terms": [
            "public safety", "crime prevention", "neighborhood safety",
            "crime rate", "crime statistics", "crime trend", "crime wave",
            "criminal activity", "crime alert"
        ],
        "weight": 2
    }
}

irrelevant_contexts = [
    "bdrms", "bthrms", "sq ft", "real estate", "property listing",
    "wildfire", "earthquake", "storm damage", "weather advisory",
    "game score", "tournament", "championship", "sports",
    "movie review", "concert", "performance", "theater",
    "stock market", "earnings report", "quarterly results",
    "school board meeting", "city council", "planning commission"
]

irrelevant_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, irrelevant_contexts)) + r")\b",
    flags=re.IGNORECASE
)

crime_justice_patterns = {
    category: re.compile(
        r"\b(" + "|".join(map(re.escape, info["terms"])) + r")\b",
        flags=re.IGNORECASE
    )
    for category, info in crime_justice_terms.items()
}

# ------------------------------------------------------------
# 2. Article Relevance Function
# ------------------------------------------------------------

def evaluate_article_relevance(text: str, min_relevance_score=3, irrelevant_penalty=2):
    total_score = 0
    matched_terms = {}
    
    irrelevant_matches = irrelevant_pattern.findall(text or "")
    
    for category_name, pattern in crime_justice_patterns.items():
        category_matches = pattern.findall(text or "")
        if category_matches:
            unique_matches = list(set(category_matches))
            weight = crime_justice_terms[category_name]["weight"]
            score = len(unique_matches) * weight
            total_score += score
            matched_terms[category_name] = unique_matches
    
    penalty = len(irrelevant_matches) * irrelevant_penalty
    total_score = max(0, total_score - penalty)
    
    is_relevant = total_score >= min_relevance_score
    
    return {
        "is_relevant": is_relevant,
        "score": total_score,
        "matches": matched_terms,
        "irrelevant_matches": irrelevant_matches
    }

# ------------------------------------------------------------
# 3. Filter News DataFrame
# ------------------------------------------------------------

def format_matches(matches: Dict[str, List[str]]) -> str:
    if not matches:
        return ""
    formatted = []
    for cat, terms in matches.items():
        formatted.append(f"{cat}: [{', '.join(terms)}]")
    return "; ".join(formatted)

def filter_news_data(df, min_score=3, irrelevant_penalty=2):
    relevance_info = df["Body"].apply(
        lambda x: evaluate_article_relevance(x, min_score, irrelevant_penalty)
    )
    
    df["is_relevant"] = relevance_info.apply(lambda x: x["is_relevant"])
    df["relevance_score"] = relevance_info.apply(lambda x: x["score"])
    df["matched_terms"] = relevance_info.apply(lambda x: format_matches(x["matches"]))
    df["irrelevant_matches"] = relevance_info.apply(
        lambda x: ", ".join(x["irrelevant_matches"])
    )
    
    df_filtered = df[df["is_relevant"]].copy()
    df_filtered = df_filtered.sort_values("relevance_score", ascending=False)
    return df_filtered

# ------------------------------------------------------------
# 4. County Extraction
# ------------------------------------------------------------

county_regex = re2.compile(
    r"\b(?:"
    r"(?:alameda|alpine|amador|butte|calaveras|colusa|contra costa|del norte|"
    r"el dorado|fresno|glenn|humboldt|imperial|inyo|kern|kings|lake|lassen|"
    r"los angeles|madera|marin|mariposa|mendocino|merced|modoc|mono|monterey|"
    r"napa|nevada|orange|placer|plumas|riverside|sacramento|san benito|"
    r"san bernardino|san diego|san francisco|san joaquin|san luis obispo|"
    r"san mateo|santa barbara|santa clara|santa cruz|shasta|sierra|siskiyou|"
    r"solano|sonoma|stanislaus|sutter|tehama|trinity|tulare|tuolumne|ventura|"
    r"yolo|yuba)(?:\s+county)?"
    r"|san francisco|los angeles"
    r")\b",
    flags=re2.IGNORECASE
)

def extract_and_count_counties(text):
    if not text:
        return dict(counties=[], counts=[])
    
    matches = county_regex.findall(text)
    cleaned = [
        re.sub(r" county$", "", m.strip().lower())
        for m in matches
    ]
    
    cleaned = [c if c not in ["sf"] else "san francisco" for c in cleaned]
    cleaned = [c if c not in ["la"] else "los angeles" for c in cleaned]
    
    counts = dict(Counter(cleaned))
    return {"counties": list(counts.keys()), "counts": list(counts.values())}

def enrich_with_counties(df):
    county_data = df["Body"].apply(extract_and_count_counties)
    df["counties"] = county_data.apply(lambda x: x["counties"])
    df["county_counts"] = county_data.apply(lambda x: x["counts"])
    
    def extract_top_counties(row):
        counties = row.get("counties", [])
        top3 = counties[:3] + [np.nan]*(3 - len(counties))
        return pd.Series(top3, index=["county1", "county2", "county3"])
    
    top_counties = county_data.apply(extract_top_counties)
    df = pd.concat([df, top_counties], axis=1)
    return df

# ------------------------------------------------------------
# 5. DA Tenure Periods
# ------------------------------------------------------------

def add_da_tenure_periods(df):
    price_start = pd.Timestamp("2023-01-01")
    boudin_start = pd.Timestamp("2020-01-01")
    boudin_recall = pd.Timestamp("2022-06-07")
    jenkins_start = pd.Timestamp("2022-07-07")

    df["Date"] = pd.to_datetime(df["Date"])

    df["during_price_tenure"] = df["Date"] >= price_start
    df["during_boudin_tenure"] = (df["Date"] >= boudin_start) & (df["Date"] < boudin_recall)
    df["during_jenkins_tenure"] = df["Date"] >= jenkins_start

    def classify_period(date):
        if pd.isna(date):
            return None
        if date < boudin_start:
            return "gascon_sf"
        elif boudin_start <= date < boudin_recall:
            return "boudin"
        elif boudin_recall <= date < jenkins_start:
            return "transition"
        else:
            return "jenkins"

    df["sf_da_period"] = df["Date"].apply(classify_period)
    return df

# ------------------------------------------------------------
# 6. Publication Categories
# ------------------------------------------------------------



def create_pub_categories(df):
    pub_categories = (
        df[["publication", "county", "bay_area"]]
        .drop_duplicates()
        .assign(pub_scope=lambda d:
            np.where(
                d["county"] == "Outside California", "National",
                np.where(d["bay_area"].notna(), "Regional", "Local")
            )
        )
    )
    return pub_categories

def merge_news_data(df, pub_categories):
    df = df.merge(pub_categories, on=["publication", "county", "bay_area"], how="left")

    for c in ["county1", "county2", "county3"]:
        df[c] = df[c].str.lower().str.strip()
    
    df["is_sf_coverage"] = df[["county1", "county2", "county3"]].apply(
        lambda x: x.str.contains("san francisco", na=False).any(), axis=1
    )
    df["is_alameda_coverage"] = df[["county1", "county2", "county3"]].apply(
        lambda x: x.str.contains("alameda", na=False).any(), axis=1
    )
    df["is_san_mateo_coverage"] = df[["county1", "county2", "county3"]].apply(
        lambda x: x.str.contains("san mateo", na=False).any(), axis=1
    )
    
    def covers_home(row):
        if row["county"] == "San Francisco" and row["is_sf_coverage"]:
            return True
        if row["county"] == "Alameda" and row["is_alameda_coverage"]:
            return True
        if row["county"] == "San Mateo" and row["is_san_mateo_coverage"]:
            return True
        if pd.notna(row["bay_area"]):
            return True
        if row["county"] == "Outside California":
            return True
        return False
    
    df["covers_home_territory"] = df.apply(covers_home, axis=1)
    return df

# ------------------------------------------------------------
# 7. Enhanced DA Name Detection - Option 2
# ------------------------------------------------------------

DA_NAME_CONFIG = {
    "boudin": {
        "patterns": [
            r"\bchesa\s+boudin\b",
            r"\bboudin\b",
            r"\bboudin's\b"
        ],
        "exclude_patterns": [
            r"boudin\s+(blanc|noir|sausage|blood)",
            r"(black|white)\s+boudin"
        ],
        "title_patterns": [
            r"(district attorney|d\.?a\.?|prosecutor)\s+(chesa\s+)?boudin",
            r"boudin,?\s+(the\s+)?(district attorney|d\.?a\.?)",
            r"(former|ex|recalled)\s+(district attorney|d\.?a\.?)\s+boudin"
        ],
        "first_name": "chesa",
        "context_boost_terms": [
            "recall", "progressive", "prosecutor",
            "crime", "sf", "san francisco", "da",
            "district attorney", "d-a", "d a", "district atty"
        ]
    },
    "jenkins": {
        "patterns": [
            r"\bbrooke\s+jenkins\b",
            r"\bjenkins\b"
        ],
        "exclude_patterns": [
            r"jenkins\s+(jr|sr|iii|iv)\b",
            r"\b(john|mary|david|robert|michael|william)\s+jenkins\b",
            r"jenkins\s+(construction|electric|plumbing|company|corp|llc|inc)\b",
            r"jenkins\s+(street|avenue|road|lane|drive|way)\b",
            r"(mr|mrs|ms|dr|prof|professor)\s+jenkins\b(?!.*\b(d\.?a\.?|district attorney))",
            r"jenkins\s+(high school|elementary|middle school|university|college)"
        ],
        "require_context": True,
        "title_patterns": [
            r"(district attorney|d\.?a\.?|prosecutor)\s+(brooke\s+)?jenkins",
            r"jenkins,?\s+(the\s+)?(district attorney|d\.?a\.?)",
            r"(interim|appointed|current)\s+(district attorney|d\.?a\.?)\s+jenkins"
        ],
        "first_name": "brooke",
        "context_boost_terms": [
            "appointed", "interim", "prosecutor",
            "crime", "sf", "san francisco", "mayor breed"
        ],
        "context_require_terms": [
            "district attorney", "d.a.", "da", "prosecutor", 
            "brooke", "sf", "san francisco", "crime", 
            "prosecution", "d-a", "d a", "district atty"
        ]
    },
    "gascon": {
        "patterns": [
            r"\bgeorge\s+gasc[oó]n\b",
            r"\bgasc[oó]n\b",
            r"\bgascon's\b"
        ],
        "exclude_patterns": [
            r"gasc[oó]n\s+(county|region|province)",
            r"(jean|pierre|marie)\s+gasc[oó]n"
        ],
        "title_patterns": [
            r"(district attorney|d\.?a\.?|prosecutor)\s+(george\s+)?gasc[oó]n",
            r"gasc[oó]n,?\s+(the\s+)?(district attorney|d\.?a\.?)",
            r"(former|ex)\s+(district attorney|d\.?a\.?)\s+gasc[oó]n"
        ],
        "first_name": "george",
        "context_boost_terms": [
            "los angeles", "la", "prosecutor", "crime", 
            "sf", "san francisco", "former", "d-a", "d a", "district atty"
        ]
    }
}


def detect_da_mentions(text, headline, da_name, config):
    da_config = config.get(da_name.lower())
    if da_config is None:
        return dict(count=0, confidence="none", positions=[], avg_score=0.0, mention_types=[])
    
    text = text or ""
    headline = headline or ""
    combined_text = f"{headline} {text}".lower()
    
    for ex in da_config.get("exclude_patterns", []):
        if re.search(ex, combined_text, flags=re.IGNORECASE):
            return dict(count=0, confidence="excluded", positions=[], avg_score=0.0, mention_types=[])
    
    mentions = []
    confidence_scores = []
    
    for pat in da_config.get("title_patterns", []):
        matches = list(re.finditer(pat, combined_text, flags=re.IGNORECASE))
        for m in matches:
            mentions.append((m.start(), "title"))
            confidence_scores.append(3.0)
    
    if da_config.get("first_name"):
        last_name_regex = da_config["patterns"][0]
        last_name_clean = re.sub(r"\\b", "", last_name_regex)
        full_name_pattern = f"\\b{da_config['first_name']}\\s+{last_name_clean}\\b"
        matches = list(re.finditer(full_name_pattern, combined_text, flags=re.IGNORECASE))
        for m in matches:
            mentions.append((m.start(), "full_name"))
            confidence_scores.append(2.5)
    
    bare_pattern = f"\\b{da_name}\\b"
    for pat in da_config.get("patterns", []):
        matches = list(re.finditer(pat, combined_text, flags=re.IGNORECASE))
        for m in matches:
            start = m.start()
            end = m.end()
            snippet = combined_text[max(0, start-100): min(len(combined_text), end+100)]
            boost = sum(bool(re.search(term, snippet, flags=re.IGNORECASE)) 
                        for term in da_config.get("context_boost_terms", []))
            confidence = 1 + boost * 0.2
            if pat == bare_pattern and da_config.get("require_context", False):
                if not any(re.search(term, snippet, flags=re.IGNORECASE) 
                           for term in da_config.get("context_require_terms", [])):
                    continue
            mentions.append((start, "bare"))
            confidence_scores.append(confidence)
    
    if not mentions:
        return dict(count=0, confidence="none", positions=[], avg_score=0.0, mention_types=[])
    
    avg_conf = np.mean(confidence_scores)
    level = "very_low"
    if avg_conf >= 2.5:
        level = "high"
    elif avg_conf >= 1.5:
        level = "medium"
    elif avg_conf >= 1.0:
        level = "low"
    
    positions = list(set([p[0] for p in mentions]))
    mention_types = list(set([p[1] for p in mentions]))
    
    return dict(
        count=len(mentions),
        confidence=level,
        positions=positions,
        avg_score=avg_conf,
        mention_types=mention_types
    )

def create_enhanced_mention_score(text, headline, da_name, config):
    result = detect_da_mentions(text, headline, da_name, config)
    weights = dict(high=1.0, medium=0.8, low=0.6, very_low=0.4, none=0, excluded=0)
    score = result["count"] * weights[result["confidence"]]
    if score > 0 and headline:
        if any(pos <= len(headline) + 1 for pos in result["positions"]):
            score *= 1.5
    return score





# Load your TSV
cleaned_news_data = pd.read_csv("RAW.tsv", sep="\t")

print(cleaned_news_data.columns.tolist())


# Run the pipeline
news_data_filtered = filter_news_data(cleaned_news_data, min_score=1)
news_data_with_counties = enrich_with_counties(news_data_filtered)
news_data_with_dates = add_da_tenure_periods(news_data_with_counties)
pub_categories = create_pub_categories(news_data_with_dates)
news_data_clean = merge_news_data(news_data_with_dates, pub_categories)

# Save result as TSV
news_data_clean.to_csv("24.07.29_complete_corpus_api_lexis_combined.tsv", sep="\t", index=False)
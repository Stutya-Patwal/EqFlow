# EqFlow Documentation

## Eqre Module - Web Scraping with BeautifulSoup

**Note:** The modules are imported as (in this example):

```from EqFlow import Eqre as eqre```
```from EqFlow import Eqre as eqai```

**Verify URL Validity:**
```eqre.verify_url(url)```

**Get BeautifulSoup Object:**
```soup = eqre.get_soup(url)```

**Get Raw Text Content:**
```raw_text = eqre.get_raw_text(soup)```

**Get Processed Text Content:**
```processed_text = eqre.get_processed_text(soup)```

**Find Specific Elements:**
```found_element = eqre.find(url, id="yKMVIe")```

## Eqai Module - AI Operations

**Tokenization:**
```tokens = eqai.tokenize_text(input_text)```

**Part-of-Speech Tagging:**
```pos_tags = eqai.pos_tagging(input_text)```

**Named Entity Recognition (NER):**
```named_entities = eqai.named_entity_recognition(input_text)```

**Sentiment Analysis:**
```sentiment = eqai.sentiment_analysis(input_text)```

**Text Summarization:**
```summary = eqai.text_summarization(input_text)```
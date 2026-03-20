from kiwipiepy import Kiwi

class KiwiTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()
        self.kiwi.add_re_rule('EF', '요$', '영', -3)
        self.kiwi.add_re_rule('EF', '요$', '용', -3)
        self.kiwi.add_re_rule('EF', '요$', '염', -3)
        self.kiwi.add_re_rule('EF', '요$', '욤', -3)
        self.kiwi.add_re_rule('EF', '야$', '얌', -3)
    
    def _merge_nouns(self, noun_tokens: list[tuple[str, int, int]]) -> list[str]:
        """
        합성어 복원
        형태소단위로 분리된 명사 토큰들을 원문의 띄어쓰기 단위로 복원합니다.
         - 예시: [('여우', 0, 2), ('주연상', 2, 3)] → ['여우주연상']
        원문의 띄어쓰기 단위로 복원합니다.
        띄어쓰기가 잘 못 된 경우에는 오류 발생 가능성 있습니다.
        """
        if not noun_tokens:
            return []

        merged = []
        current_form, current_start, current_len = noun_tokens[0]

        for form, start, length in noun_tokens[1:]:
            if current_start + current_len == start:
                # 붙어있음 → 합성
                current_form = f"{current_form}{form}"
                current_len = (start + length) - current_start
            else:
                # 띄어있음 → 별개의 명사로 간주
                merged.append(current_form)
                current_form, current_start, current_len = form, start, length

        merged.append(current_form)
        return merged
    
    def extract_nouns(self, corpus:str) -> list[str]:
        """
        corpus에서 명사 추출(띄어쓰기로 단어 구분)
        """
        tokens = self.kiwi.tokenize(corpus)
        processed_nouns = []
        noun = []
        
        for token in tokens:
            if "N" in token.tag:
                if token.tag == "ETN": # 명사형 전성 어미일 경우 처리 X
                    pass
                elif token.tag == "XPN": # 체언 접두어일 경우 처리 X
                    pass
                elif token.tag == "NNB": # 의존 명사 처리 X
                    pass
                elif token.tag == "NP": # 대명사 처리 X
                    pass
                elif token.tag == "XSN": # 명사 파생 접미사 처리 X
                    pass
                elif token.form[-1] == "." or token.form[-1] == ",":
                    noun.append((token.form.strip(), token.start, token.len))

                else:
                    noun.append((token.form.strip(), token.start, token.len))
            elif token.tag in ["SH", "SL", "SW"]: # 외국어 및 특수 문자 처리
                noun.append((token.form.strip(), token.start, token.len))

            else:
                if len(noun)>0:
                    processed_nouns.extend(self._merge_nouns(noun))
                noun = []
        if len(noun)>0:
            processed_nouns.extend(self._merge_nouns(noun))
            
        return processed_nouns
    
    def extract_nouns_with_ngram(self, corpus: str, n: int = 2) -> list[str]:
        nouns = self.extract_nouns(corpus)
        
        # 원본 명사 + bigram 합성
        ngrams = []
        for i in range(len(nouns) - n + 1):
            ngrams.append("".join(nouns[i:i+n]))
        
        return nouns + ngrams
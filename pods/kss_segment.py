# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2021 Heewon Jeon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Optional
from jina.executors.segmenters import BaseSegmenter
from jina.executors.decorators import single

from kss import split_sentences

class KSSSentencizer(BaseSegmenter):
    """
    Segment text into sentences using :class:`nltk.PunktSentenceTokenizer`
    for the specified language.
    Example:
    >>> sentencizer = KSSSentencizer() 
    >>> text = "Today is a good day. Can't wait for tomorrow!"
    >>> sentencizer.segment([text])
    [{'text': 'Today is a good day.', 'offset': 0, 'location': [0, 20]}, {'text': "Can't wait for tomorrow!", 'offset': 1, 'location': [21, 45]}]
    :param safe: default=True. Optional parameter,If your article follows the punctuation rules relatively well, set the safe=True. (default is False) 
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

   
    def __init__(
        self,
        safe=True,
        *args,
        **kwargs,
    ):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.safe = safe


    @single
    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Segment text into sentences.
        :param text: The text to be sentencized.
        :type text: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: List of dictonaries representing sentences with three keys: ``text``, for representing the text of the sentence; ``offset``, representing the order of the sentence within input text; ``location``, a list with start and end indeces for the sentence.
        :rtype: List[Dict]
        """
        sentences = split_sentences(text, safe=self.safe)
        results = []
        start = 0
        for i, s in enumerate(sentences):
            start = text[start:].find(s) + start
            end = start + len(s)
            results.append({'text': s, 'offset': i, 'location': [start, end]})
            start = end

        return results


if __name__ == '__main__':
    kss = KSSSentencizer()

    print(kss.segment('미리 예약을 할 수 있는 시스템으로 합리적인 가격에 여러 종류의 생선, 그리고 다양한 부위를 즐길 수 있기 때문이다. 계절에 따라 모둠회의 종류는 조금씩 달라지지만 자주 올려주는 참돔 마스까와는 특히 맛이 매우 좋다. 일반 모둠회도 좋지만 좀 더 특별한 맛을 즐기고 싶다면 특수 부위 모둠회를 추천한다 제철 생선 5~6가지 구성에 평소 접하지 못했던 부위까지 색다르게 즐길 수 있다.'))
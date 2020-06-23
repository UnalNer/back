def recognize_entities(text):
  return [{
      'text': 'ner1',
      'type': 'MEDICINE'
  }, {
      'text': 'ner2',
      'type': 'ADJ'
  }, {
      'text': 'ner3',
      'type': 'INSTITUTION'
  }]

print(recognize_entities(''))
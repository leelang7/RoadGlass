// 예) 상세 우측 패널 안
const kApiBase = 'http://ec2-54-67-48-0.us-west-1.compute.amazonaws.com'; // 본인 호스트

ElevatedButton.icon(
  icon: const Icon(Icons.image),
  label: const Text('원본/오버레이 보기'),
  onPressed: () {
    final recId = selectedRecord.id; // <- 너가 들고있는 선택된 레코드의 id
    showAnalysisImages(
      context,
      baseUrl: kApiBase,
      id: recId,
    );
  },
)

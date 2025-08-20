document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const messagesContainer = document.querySelector('.messages');
    const newChatBtn = document.querySelector('.new-chat-btn');

    if (chatForm) {
        chatForm.addEventListener('submit', function(event) {
            event.preventDefault();
            sendMessage();
        });
    }

    document.getElementById("chat-input").addEventListener("keydown", function(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    newChatBtn.addEventListener('click', () => {
        messagesContainer.innerHTML = '<div class="message bot"><p>안녕하세요! 저는 캠토리입니다. 캠핑에 관한 모든 질문에 답해드릴게요! 어떤 도움이 필요하신가요? 😊</p></div>';
        clearMap();
        document.getElementById('route-list').innerHTML = '';
        document.getElementById('map-and-list-container').style.display = 'none';
        stopSpeech(); // 새 대화 시작 시 음성 중지
    });
});

function stopSpeech() {
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
}

function speak(text) {
    stopSpeech(); // 새 음성 재생 전 기존 음성 중지
    if ('speechSynthesis' in window) {
        // HTML 태그 및 줄바꿈 제거
        const cleanText = text.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '');
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'ko-KR'; // 한국어 설정
        utterance.rate = 1.1; // 약간 빠른 속도
        utterance.pitch = 1; // 기본 톤
        window.speechSynthesis.speak(utterance);
    } else {
        console.log('이 브라우저는 음성 합성을 지원하지 않습니다.');
    }
}

function sendMessage() {
    const userInput = document.getElementById("chat-input");
    const message = userInput.value.trim();
    if (!message) return;

    stopSpeech(); // 사용자가 메시지를 보내면 이전 봇 응답 음성 중지

    const chatBox = document.querySelector(".messages");
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user';
    userMessageDiv.innerHTML = `<p>${message}</p>`;
    chatBox.appendChild(userMessageDiv);
    userInput.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    const botResponseDiv = document.createElement('div');
    botResponseDiv.className = 'message bot';
    const botParagraph = document.createElement('p');
    botParagraph.innerHTML = '답변 생성 중...';
    botResponseDiv.appendChild(botParagraph);
    chatBox.appendChild(botResponseDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    const markers = new Array();

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    })
    .then(response => response.json())
    .then(data => {
        const fullTextForDisplay = data.answer.replace(/\n/g, '<br>');
        botParagraph.innerHTML = '';
        let i = 0;
        const speed = 30;

        function typeWriter() {
            if (i < fullTextForDisplay.length) {
                if (fullTextForDisplay.substring(i, i + 4) === '<br>') {
                    botParagraph.innerHTML += '<br>';
                    i += 4;
                } else {
                    botParagraph.innerHTML += fullTextForDisplay.charAt(i);
                    i++;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
                setTimeout(typeWriter, speed);
            } 
            // 전체 답변에 대한 자동 TTS는 제거
        }

        clearMap();

        const routeList = document.getElementById("route-list");
        routeList.innerHTML = '';
        
        const mapAndListContainer = document.getElementById('map-and-list-container');

        if (data.locations && data.locations.length > 0) {
            mapAndListContainer.style.display = 'block';

            data.locations.forEach((location, index) => {
                const listItem = document.createElement("li");
                listItem.id = `location-${index}`;
                listItem.innerHTML = `<strong>${location.name}</strong><br>주소: ${location.address}`;
                
                // 클릭 이벤트에 설명(description)과 위도/경도 전달
                listItem.onclick = () => {
                    console.log("[DEBUG] Location clicked:", location);
                    console.log("[DEBUG] Description to speak:", location.local_meta); //location.description

                    // 지도 중앙 이동 및 경로 그리기
                    drawRoute(location.latitude, location.longitude, location.name, location.address, listItem.id);
                    
                    // 선택된 장소의 설명만 TTS로 재생 (데이터 확인 및 폴백 로직 추가)
                    if (location.description && location.description.trim() !== '') {
                        console.log("[DEBUG] Calling speak() with description.");
                        speak(location.description);
                    } else {
                        console.log("[DEBUG] Description is empty. Speaking name and address as fallback.");
                        const fallbackText = `${location.name}. 주소: ${location.address}`;
                        speak(fallbackText);
                    }

                    // 지도 중앙으로 이동
                    map.setCenter(new Tmapv2.LatLng(location.latitude, location.longitude));
                    map.setZoom(15);
                };
                routeList.appendChild(listItem);

                if (location.latitude && location.longitude) {
                    const markerPosition = new Tmapv2.LatLng(location.latitude, location.longitude);
                    const marker = new Tmapv2.Marker({
                        position: markerPosition,
                        map: map,
                        label: location.name,
                        icon: Tmapv2.asset.Icon.get(`b_m_${index+1}`)
                    });
                    markers.push(marker);
                }
            });

            const firstLocation = data.locations[0];
            if (firstLocation.latitude && firstLocation.longitude) {
                map.setCenter(new Tmapv2.LatLng(firstLocation.latitude, firstLocation.longitude));
            }

            if (map) {
                map.resize();
            }

        } else {
            mapAndListContainer.style.display = 'none';
        }

        typeWriter();
    })
    .catch(error => {
        console.error('Error:', error);
        botParagraph.innerHTML = `오류가 발생했습니다. 다시 시도해주세요.`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}


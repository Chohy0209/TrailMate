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
    });
});

function sendMessage() {
    const userInput = document.getElementById("chat-input");
    const message = userInput.value.trim();
    if (!message) return;

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
        const fullText = data.answer.replace(/\n/g, '<br>');
        botParagraph.innerHTML = '';
        let i = 0;
        const speed = 30;

        function typeWriter() {
            if (i < fullText.length) {
                if (fullText.substring(i, i + 4) === '<br>') {
                    botParagraph.innerHTML += '<br>';
                    i += 4;
                } else {
                    botParagraph.innerHTML += fullText.charAt(i);
                    i++;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
                setTimeout(typeWriter, speed);
            }
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
                listItem.onclick = () => drawRoute(location.latitude, location.longitude, location.name, listItem.id);
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


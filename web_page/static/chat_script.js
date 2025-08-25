const ChatApp = {
    config: { tmapApiKey: '', weatherApiKey: '', initialMessage: '' },
    state: { userLocation: null, map: null, markers: [], polylines: [] },
    elements: {},

    init(config) {
        this.config = { ...this.config, ...config };
        this.cacheDOMElements();
        this.addEventListeners();
        this.initTheme();
        this.initTmap();
        if (this.config.initialMessage) {
            this.elements.chatInput.value = this.config.initialMessage;
            this.sendMessage();
        }
    },

    cacheDOMElements() {
        this.elements = {
            chatForm: document.getElementById('chat-form'), chatInput: document.getElementById('chat-input'),
            messagesContainer: document.querySelector('.messages'), newChatBtn: document.querySelector('.new-chat-btn'),
            routeList: document.getElementById('route-list'), mapAndListContainer: document.getElementById('map-and-list-container'),
            weatherInfo: document.getElementById('weather-info'), body: document.body,
            menuToggleBtn: document.getElementById('menu-toggle-btn'), sidebar: document.querySelector('.sidebar'),
            closeSidebarBtn: document.querySelector('.close-sidebar-btn'),
            darkModeSwitchMobile: document.getElementById('dark-mode-switch-mobile'),
            darkModeSwitchDesktop: document.getElementById('dark-mode-switch-desktop'),
        };
    },

    addEventListeners() {
        this.elements.chatForm.addEventListener('submit', e => { e.preventDefault(); this.sendMessage(); });
        this.elements.chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }});
        this.elements.newChatBtn.addEventListener('click', () => this.resetChat());
        this.elements.menuToggleBtn.addEventListener('click', () => this.toggleSidebar());
        this.elements.closeSidebarBtn.addEventListener('click', () => this.closeSidebar());
        document.addEventListener('click', (e) => {
            if (!this.elements.sidebar.classList.contains('open')) return;
            const isClickInsideSidebar = this.elements.sidebar.contains(e.target);
            const isClickOnMenuBtn = this.elements.menuToggleBtn.contains(e.target);
            if (!isClickInsideSidebar && !isClickOnMenuBtn) this.closeSidebar();
        });
        this.elements.darkModeSwitchMobile.addEventListener('change', () => this.syncAndToggleTheme(this.elements.darkModeSwitchMobile));
        this.elements.darkModeSwitchDesktop.addEventListener('change', () => this.syncAndToggleTheme(this.elements.darkModeSwitchDesktop));
    },
    
    toggleSidebar() { this.elements.sidebar.classList.toggle('open'); },
    closeSidebar() { this.elements.sidebar.classList.remove('open'); },
    syncAndToggleTheme(masterSwitch) {
        const isChecked = masterSwitch.checked;
        if (this.elements.darkModeSwitchMobile.checked !== isChecked) this.elements.darkModeSwitchMobile.checked = isChecked;
        if (this.elements.darkModeSwitchDesktop.checked !== isChecked) this.elements.darkModeSwitchDesktop.checked = isChecked;
        this.toggleTheme();
    },
    initTheme() { this.setTheme(localStorage.getItem('theme') || 'dark'); },
    setTheme(theme) {
        const isDark = theme === 'dark';
        this.elements.body.classList.toggle('dark-mode', isDark);
        this.elements.darkModeSwitchMobile.checked = isDark;
        this.elements.darkModeSwitchDesktop.checked = isDark;
    },
    toggleTheme() {
        const newTheme = this.elements.darkModeSwitchDesktop.checked ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        this.setTheme(newTheme);
    },

    sendMessage() {
        const message = this.elements.chatInput.value.trim();
        if (!message) return;
        this.stopSpeech();
        this.displayUserMessage(message);
        this.elements.chatInput.value = '';
        const botTypingDiv = this.displayBotTypingIndicator();

        fetch("/chat", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            const fullTextForDisplay = data.answer.replace(/\n/g, '<br>');
            this.displayBotMessage(fullTextForDisplay, botTypingDiv);
            this.clearMap();

            if (data && data.locations && Array.isArray(data.locations) && data.locations.length > 0) {
                this.elements.mapAndListContainer.style.display = 'flex';
                this.displayLocations(data.locations);
            } else {
                this.elements.mapAndListContainer.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('채팅 처리 중 오류:', error);
            this.displayBotMessage('오류가 발생했습니다. 다시 시도해주세요.', botTypingDiv);
        });
    },

    displayUserMessage(message) {
        const html = `<div class="message user"><p>${message}</p></div>`;
        this.elements.messagesContainer.insertAdjacentHTML('beforeend', html);
        this.scrollToBottom();
    },
    displayBotTypingIndicator() {
        const botResponseDiv = document.createElement('div');
        botResponseDiv.className = 'message bot';
        botResponseDiv.innerHTML = `<p>답변 생성 중...</p>`;
        this.elements.messagesContainer.appendChild(botResponseDiv);
        this.scrollToBottom();
        return botResponseDiv;
    },
    displayBotMessage(htmlContent, containerDiv) {
        const botParagraph = containerDiv.querySelector('p');
        botParagraph.innerHTML = '';
        this.typeWriter(botParagraph, htmlContent);
    },
    typeWriter(element, text, i = 0, speed = 20) {
        if (i < text.length) {
            if (text.charAt(i) === '<') {
                const endOfTag = text.indexOf('>', i);
                if (endOfTag !== -1) {
                    element.innerHTML += text.substring(i, endOfTag + 1);
                    i = endOfTag;
                }
            } else {
                element.innerHTML += text.charAt(i);
            }
            i++;
            this.scrollToBottom();
            setTimeout(() => this.typeWriter(element, text, i, speed), speed);
        }
    },
    resetChat() {
        this.elements.messagesContainer.innerHTML = '<div class="message bot"><p>안녕하세요! 캠토리입니다. 무엇을 도와드릴까요? 😊</p></div>';
        this.clearMap();
        this.elements.mapAndListContainer.style.display = 'none';
        this.stopSpeech();
    },
    scrollToBottom() { this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight; },
    stopSpeech() { if (window.speechSynthesis?.speaking) window.speechSynthesis.cancel(); },
    speak(text) {
        this.stopSpeech();
        const cleanText = text.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '');
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'ko-KR'; utterance.rate = 1.1;
        window.speechSynthesis.speak(utterance);
    },

    initTmap() {
        this.state.map = new Tmapv2.Map("map_div", {
            center: new Tmapv2.LatLng(37.5665, 126.9780), width: "100%", height: "100%", zoom: 12,
            fontFace: "#000000" 
        });
        this.getWeatherAndDisplay(37.5665, 126.9780);
        this.getUserLocation();
    },
    getUserLocation() {
        const seoulStation = { lat: 37.5547, lon: 126.9704 };
        const setLocation = (loc, label, icon) => {
            this.state.userLocation = loc;
            const latLng = new Tmapv2.LatLng(loc.lat, loc.lon);
            this.state.map.setCenter(latLng);
            this.addMarker(latLng, label, icon);
        };
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                pos => setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude }, '현재 위치', 'b_m_a'),
                () => setLocation(seoulStation, '기본 위치 (서울역)', 'b_m_b')
            );
        } else {
            setLocation(seoulStation, '기본 위치 (서울역)', 'b_m_b');
        }
    },
    displayLocations(locations) {
        this.elements.routeList.innerHTML = '';
        locations.forEach((location, index) => {
            const listItem = this.createLocationListItem(location, index);
            this.elements.routeList.appendChild(listItem);
            if (location.latitude && location.longitude) {
                const position = new Tmapv2.LatLng(location.latitude, location.longitude);
                this.addMarker(position, location.name, `b_m_${index + 1}`);
            }
        });
        const firstLocation = locations[0];
        if (firstLocation?.latitude && firstLocation?.longitude) {
            this.state.map.setCenter(new Tmapv2.LatLng(firstLocation.latitude, firstLocation.longitude));
        }
        if (this.state.map) this.state.map.resize();
    },
    createLocationListItem(location, index) {
        const listItem = document.createElement("li");
        listItem.id = `location-${index}`;
        listItem.innerHTML = `<strong>${location.name}</strong><br>주소: ${location.address}`;
        listItem.onclick = () => {
            this.drawRoute(location.latitude, location.longitude, location.name, location.address, listItem.id);
            this.speak(`${location.name}. 주소는 ${location.address} 입니다.`);
            this.state.map.setCenter(new Tmapv2.LatLng(location.latitude, location.longitude));
        };
        return listItem;
    },
    addMarker(position, label, iconType) {
        const marker = new Tmapv2.Marker({
            position,
            map: this.state.map,
            label,
            icon: Tmapv2.asset.Icon.get(iconType),
            // labelStyle: {
            //     color: "#000000",
            //     fontWeight: "bold",
            //     // textShadow: "1px 1px 3px #FFFFFF, -1px -1px 3px #FFFFFF, 1px -1px 3px #FFFFFF, -1px 1px 3px #FFFFFF"
            // }
        });
        this.state.markers.push(marker);
    },
    clearMap() {
        this.state.markers.forEach(m => m.setMap(null));
        this.state.polylines.forEach(p => p.setMap(null));
        this.state.markers = []; this.state.polylines = [];
        this.elements.routeList.innerHTML = '';
    },
    drawRoute(endLat, endLon, endName, endAddress, listItemId) {
        this.state.polylines.forEach(p => p.setMap(null));
        this.state.polylines = [];
        this.getWeatherAndDisplay(endLat, endLon);
        if (!this.state.userLocation) { alert("현재 위치 정보가 없습니다."); return; }
        fetch("/get_tmap_route", {
            method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify({
                startX: this.state.userLocation.lon, startY: this.state.userLocation.lat,
                endX: endLon, endY: endLat, endName: endName
            })
        })
        .then(response => response.json()).then(data => {
            if (data.error) { console.error("TMAP Route Error:", data.error); return; }
            if (data.features?.length > 0) {
                this.drawRoutePolylinesWithTraffic(data.features);
                this.updateRouteInfo(data.features[0].properties, endName, endAddress, listItemId);
            }
        }).catch(error => console.error('Error fetching TMAP route:', error));
    },
    drawRoutePolylinesWithTraffic(features) {
        const trafficColors = { "1": "#009900", "2": "#FFD400", "3": "#FF8C00", "4": "#FF0000" };
        features.forEach(feature => {
            if (feature.geometry.type === "LineString" && feature.geometry.traffic) {
                feature.geometry.traffic.forEach(([from, to, congestion]) => {
                    const path = feature.geometry.coordinates.slice(from, to + 1).map(c => new Tmapv2.LatLng(c[1], c[0]));
                    if (path.length < 2) return;
                    const polyline = new Tmapv2.Polyline({ path, strokeColor: trafficColors[congestion] || "#808080", strokeWeight: 8, map: this.state.map });
                    this.state.polylines.push(polyline);
                });
            }
        });
    },
    updateRouteInfo(properties, endName, endAddress, listItemId) {
        const listItem = document.getElementById(listItemId);
        if (listItem && properties) {
            const time = Math.round(properties.totalTime / 60);
            const toll = (properties.totalFare || 0).toLocaleString('ko-KR');
            const dist = (properties.totalDistance / 1000).toFixed(1);
            listItem.innerHTML = `<strong>${endName}</strong><br>주소: ${endAddress}<hr>거리: ${dist}km, 시간: 약 ${time}분, 통행료: ${toll}원`;
        }
    },
    getWeatherAndDisplay(lat, lon) {
        if (!this.config.weatherApiKey) return;
        const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${this.config.weatherApiKey}&units=metric&lang=kr`;
        fetch(url).then(r => r.json()).then(data => {
            if (data.list?.length > 0) this.elements.weatherInfo.innerHTML = this.formatWeatherForecast(data);
        }).catch(e => console.error('Weather fetch error:', e));
    },
    formatWeatherForecast(data) {
        let html = `<h3>${data.city.name} 날씨 예보</h3><div class="weather-forecast-container">`;
        const daily = {};
        data.list.forEach(item => {
            const day = new Date(item.dt * 1000).toLocaleString('ko-KR', { weekday: 'short' });
            if (!daily[day]) daily[day] = { temps: [], weathers: [], icons: [] };
            daily[day].temps.push(item.main.temp);
            daily[day].weathers.push(item.weather[0].description);
            daily[day].icons.push(item.weather[0].icon);
        });
        for (const day in daily) {
            const d = daily[day];
            const avgTemp = (d.temps.reduce((a, b) => a + b, 0) / d.temps.length).toFixed(1);
            const mostCommon = arr => arr.sort((a,b) => arr.filter(v=>v===a).length - arr.filter(v=>v===b).length).pop();
            html += `<div class="weather-day">
                <strong class="weather-day-text">${day}</strong>
                <img src="http://openweathermap.org/img/w/${mostCommon(d.icons)}.png" alt="${mostCommon(d.weathers)}">
                <span class="weather-desc">${mostCommon(d.weathers)}</span>
                <span class="weather-temp">${avgTemp}°C</span>
            </div>`;
        }
        return html + `</div>`;
    }
};

// // static/chat_script.js
// const ChatApp = {
//     // 상태 및 설정
//     config: { tmapApiKey: '', weatherApiKey: '', initialMessage: '' },
//     state: { userLocation: null, map: null, markers: [], polylines: [] },
//     elements: {},

//     // 초기화
//     init(config) {
//         this.config = { ...this.config, ...config };
//         this.cacheDOMElements();
//         this.addEventListeners();
//         this.initTheme();
//         this.initTmap();
//         if (this.config.initialMessage) {
//             this.elements.chatInput.value = this.config.initialMessage;
//             this.sendMessage();
//         }
//     },

//     cacheDOMElements() {
//         this.elements = {
//             chatForm: document.getElementById('chat-form'), chatInput: document.getElementById('chat-input'),
//             messagesContainer: document.querySelector('.messages'), newChatBtn: document.querySelector('.new-chat-btn'),
//             routeList: document.getElementById('route-list'), mapAndListContainer: document.getElementById('map-and-list-container'),
//             weatherInfo: document.getElementById('weather-info'), body: document.body,
//             menuToggleBtn: document.getElementById('menu-toggle-btn'),
//             sidebar: document.querySelector('.sidebar'),
//             closeSidebarBtn: document.querySelector('.close-sidebar-btn'), // ✨ 이 부분이 버튼을 찾습니다
//             darkModeSwitchMobile: document.getElementById('dark-mode-switch-mobile'),
//             darkModeSwitchDesktop: document.getElementById('dark-mode-switch-desktop'),
//         };
//     },

//     addEventListeners() {
//         this.elements.chatForm.addEventListener('submit', e => { e.preventDefault(); this.sendMessage(); });
//         this.elements.chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }});
//         this.elements.newChatBtn.addEventListener('click', () => this.resetChat());
//         this.elements.menuToggleBtn.addEventListener('click', () => this.toggleSidebar());
//         this.elements.darkModeSwitchMobile.addEventListener('change', () => this.syncAndToggleTheme(this.elements.darkModeSwitchMobile));
//         this.elements.darkModeSwitchDesktop.addEventListener('change', () => this.syncAndToggleTheme(this.elements.darkModeSwitchDesktop));

//         // ✨ (수정) 버튼이 정상적으로 찾아졌는지 확인 후 이벤트를 추가하는 방어 코드
//         if (this.elements.closeSidebarBtn) 
//         {
//             this.elements.closeSidebarBtn.addEventListener('click', () => this.closeSidebar());
//         } 
//         else 
//         {
//             console.error('오류: 사이드바 닫기 버튼을 찾을 수 없습니다.');
//         }

//         document.addEventListener('click', (e) => {
//             if (!this.elements.sidebar.classList.contains('open')) 
//             {
//                     return;
//             }
//             const isClickInsideSidebar = this.elements.sidebar.contains(e.target);
//             const isClickOnMenuBtn = this.elements.menuToggleBtn.contains(e.target);
//             if (!isClickInsideSidebar && !isClickOnMenuBtn) 
//             {
//                 this.closeSidebar();
//             }
//         });
//     },

//     toggleSidebar() { this.elements.sidebar.classList.toggle('open'); },

//     closeSidebar() { this.elements.sidebar.classList.remove('open'); },
    
//     syncAndToggleTheme(masterSwitch) {
//         const isChecked = masterSwitch.checked;
//         if (this.elements.darkModeSwitchMobile.checked !== isChecked) this.elements.darkModeSwitchMobile.checked = isChecked;
//         if (this.elements.darkModeSwitchDesktop.checked !== isChecked) this.elements.darkModeSwitchDesktop.checked = isChecked;
//         this.toggleTheme();
//     },

//     initTheme() { this.setTheme(localStorage.getItem('theme') || 'dark'); },
//     setTheme(theme) {
//         const isDark = theme === 'dark';
//         this.elements.body.classList.toggle('dark-mode', isDark);
//         this.elements.darkModeSwitchMobile.checked = isDark;
//         this.elements.darkModeSwitchDesktop.checked = isDark;
//     },
//     toggleTheme() {
//         const newTheme = this.elements.darkModeSwitchDesktop.checked ? 'dark' : 'light';
//         localStorage.setItem('theme', newTheme);
//         this.setTheme(newTheme);
//     },

//     // 채팅 기능
//     sendMessage() {
//         const message = this.elements.chatInput.value.trim();
//         if (!message) return;
//         this.stopSpeech();
//         this.displayUserMessage(message);
//         this.elements.chatInput.value = '';
//         const botTypingDiv = this.displayBotTypingIndicator();

//         fetch("/chat", {
//             method: "POST", headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ message })
//         })
//         .then(response => response.json())
//         .then(data => {
//             // ✨ 디버깅을 위해 서버로부터 받은 데이터를 콘솔에 출력합니다.
//             console.log("서버로부터 받은 응답:", data);

//             const fullTextForDisplay = data.answer.replace(/\n/g, '<br>');
//             this.displayBotMessage(fullTextForDisplay, botTypingDiv);
            
//             this.clearMap();

//             // ✨ if 조건문을 더 명확하게 수정하고, 콘솔 로그를 추가합니다.
//             if (data && data.locations && Array.isArray(data.locations) && data.locations.length > 0) {
//                 console.log(`${data.locations.length}개의 장소를 표시합니다.`);
//                 this.elements.mapAndListContainer.style.display = 'flex';
//                 this.displayLocations(data.locations);
//             } else {
//                 console.log("표시할 장소 정보가 없습니다. 리스트를 숨깁니다.");
//                 this.elements.mapAndListContainer.style.display = 'none';
//             }
//         })
//         .catch(error => {
//             console.error('채팅 처리 중 오류:', error);
//             this.displayBotMessage('오류가 발생했습니다. 다시 시도해주세요.', botTypingDiv);
//         });
//     },

//     displayUserMessage(message) {
//         const html = `<div class="message user"><p>${message}</p></div>`;
//         this.elements.messagesContainer.insertAdjacentHTML('beforeend', html);
//         this.scrollToBottom();
//     },

//     displayBotTypingIndicator() {
//         const botResponseDiv = document.createElement('div');
//         botResponseDiv.className = 'message bot';
//         botResponseDiv.innerHTML = `<p>답변 생성 중...</p>`;
//         this.elements.messagesContainer.appendChild(botResponseDiv);
//         this.scrollToBottom();
//         return botResponseDiv;
//     },

//     displayBotMessage(htmlContent, containerDiv) {
//         const botParagraph = containerDiv.querySelector('p');
//         botParagraph.innerHTML = '';
//         this.typeWriter(botParagraph, htmlContent);
//     },

//     handleBotResponse(data, containerDiv) {
//         const fullTextForDisplay = data.answer.replace(/\n/g, '<br>');
//         this.displayBotMessage(fullTextForDisplay, containerDiv);
//         this.clearMap();
        
//         if (data.locations && data.locations.length > 0) {
//             this.elements.mapAndListContainer.style.display = 'flex';
//             this.displayLocations(data.locations);
//         } else {
//             this.elements.mapAndListContainer.style.display = 'none';
//         }
//     },

//     // ✨ typeWriter 함수를 아래 코드로 교체해주세요.
//     typeWriter(element, text, i = 0, speed = 20) {
//         if (i < text.length) {
//             // ✨ 현재 문자가 '<'인지 확인하여 태그의 시작을 감지합니다.
//             if (text.charAt(i) === '<') {
//                 // '>' 문자를 찾아 태그의 끝을 확인합니다.
//                 const endOfTag = text.indexOf('>', i);
                
//                 if (endOfTag !== -1) {
//                     // '<'부터 '>'까지의 전체 태그 문자열(예: '<br>')을 한 번에 추가합니다.
//                     const tag = text.substring(i, endOfTag + 1);
//                     element.innerHTML += tag;
//                     // 인덱스(i)를 태그가 끝난 위치로 바로 이동시킵니다.
//                     i = endOfTag;
//                 }
//             } else {
//                 // 일반 텍스트는 기존처럼 한 글자씩 추가합니다.
//                 element.innerHTML += text.charAt(i);
//             }
            
//             i++; // 다음 문자로 인덱스를 이동합니다.
//             this.scrollToBottom();
//             setTimeout(() => this.typeWriter(element, text, i, speed), speed);
//         }
//     },

//     resetChat() {
//         this.elements.messagesContainer.innerHTML = '<div class="message bot"><p>안녕하세요! 캠토리입니다. 무엇을 도와드릴까요? 😊</p></div>';
//         this.clearMap();
//         this.elements.mapAndListContainer.style.display = 'none';
//         this.stopSpeech();
//     },

//     scrollToBottom() { this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight; },

//     // 음성 합성 (TTS)
//     stopSpeech() { if (window.speechSynthesis?.speaking) window.speechSynthesis.cancel(); },
//     speak(text) {
//         this.stopSpeech();
//         const cleanText = text.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '');
//         const utterance = new SpeechSynthesisUtterance(cleanText);
//         utterance.lang = 'ko-KR'; utterance.rate = 1.1;
//         window.speechSynthesis.speak(utterance);
//     },

//     // TMAP 및 위치
//     initTmap() {
//         this.state.map = new Tmapv2.Map("map_div", {
//             center: new Tmapv2.LatLng(37.5665, 126.9780), width: "100%", height: "100%", zoom: 12
//         });
//         this.getWeatherAndDisplay(37.5665, 126.9780);
//         this.getUserLocation();
//     },

//     getUserLocation() {
//         const seoulStation = { lat: 37.5547, lon: 126.9704 };
//         const setLocation = (loc, label, icon) => {
//             this.state.userLocation = loc;
//             const latLng = new Tmapv2.LatLng(loc.lat, loc.lon);
//             this.state.map.setCenter(latLng);
//             this.addMarker(latLng, label, icon);
//         };

//         if (navigator.geolocation) {
//             navigator.geolocation.getCurrentPosition(
//                 pos => setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude }, '현재 위치', 'b_m_a'),
//                 () => {
//                     console.warn("위치 정보 실패. 기본 위치(서울역)로 설정합니다.");
//                     setLocation(seoulStation, '기본 위치 (서울역)', 'b_m_b');
//                 }
//             );
//         } else {
//             console.error("위치 정보 서비스를 지원하지 않습니다.");
//             setLocation(seoulStation, '기본 위치 (서울역)', 'b_m_b');
//         }
//     },

//     displayLocations(locations) {
//         // 기존 리스트를 초기화합니다.
//         this.elements.routeList.innerHTML = '';

//         // 서버에서 받은 locations 배열을 순회합니다.
//         locations.forEach((location, index) => {
//             // 각 장소에 대한 리스트 아이템(li)을 생성합니다.
//             const listItem = this.createLocationListItem(location, index);
//             // 생성된 아이템을 화면의 리스트(ul)에 추가합니다.
//             this.elements.routeList.appendChild(listItem);

//             // 해당 장소의 위도, 경도 정보가 있으면 지도에 마커를 추가합니다.
//             if (location.latitude && location.longitude) {
//                 const position = new Tmapv2.LatLng(location.latitude, location.longitude);
//                 this.addMarker(position, location.name, `b_m_${index + 1}`);
//             }
//         });

//         // 첫 번째 장소를 지도의 중심으로 설정합니다.
//         const firstLocation = locations[0];
//         if (firstLocation && firstLocation.latitude && firstLocation.longitude) {
//             this.state.map.setCenter(new Tmapv2.LatLng(firstLocation.latitude, firstLocation.longitude));
//         }
        
//         // 지도 크기를 컨테이너에 맞게 다시 조정합니다.
//         this.state.map.resize();
//     },

//     createLocationListItem(location, index) {
//         const listItem = document.createElement("li");
//         listItem.id = `location-${index}`;
//         listItem.innerHTML = `<strong>${location.name}</strong><br>주소: ${location.address}`;
//         listItem.onclick = () => {
//             this.drawRoute(location.latitude, location.longitude, location.name, location.address, listItem.id);
//             const textToSpeak = `${location.name}. 주소는 ${location.address} 입니다.`;
//             // this.speak(textToSpeak);
//             this.state.map.setCenter(new Tmapv2.LatLng(location.latitude, location.longitude));
//         };
//         return listItem;
//     },

//     addMarker(position, label, iconType) {
//         const marker = new Tmapv2.Marker({
//             position,
//             map: this.state.map,
//             label,
//             icon: Tmapv2.asset.Icon.get(iconType),
//             // ✨ 지도 위 글씨를 검정색으로 설정하고 가독성을 위한 그림자 추가
//             labelStyle: {
//                 color: "#000000", // 글자색: 검정
//                 fontWeight: "bold",
//                 // 흰색 테두리 효과를 주어 어떤 지도 배경에서도 잘 보이게 함
//                 textShadow: "1px 1px 3px #FFFFFF, -1px -1px 3px #FFFFFF, 1px -1px 3px #FFFFFF, -1px 1px 3px #FFFFFF"
//             }
//         });
//         this.state.markers.push(marker);
//     },

//     clearMap() {
//         this.state.markers.forEach(m => m.setMap(null));
//         this.state.polylines.forEach(p => p.setMap(null));
//         this.state.markers = [];
//         this.state.polylines = [];
//         this.elements.routeList.innerHTML = '';
//     },

//     drawRoute(endLat, endLon, endName, endAddress, listItemId) {
//         this.state.polylines.forEach(p => p.setMap(null));
//         this.state.polylines = [];
//         this.getWeatherAndDisplay(endLat, endLon);
//         if (!this.state.userLocation) { alert("현재 위치 정보가 없습니다."); return; }

//         fetch("/get_tmap_route", {
//             method: "POST", headers: { "content-type": "application/json" },
//             body: JSON.stringify({
//                 startX: this.state.userLocation.lon, startY: this.state.userLocation.lat,
//                 endX: endLon, endY: endLat, endName: endName
//             })
//         })
//         .then(response => response.json())
//         .then(data => {
//             if (data.error) { console.error("TMAP Route Error:", data.error); return; }
//             if (data.features?.length > 0) {
//                 this.drawRoutePolylinesWithTraffic(data.features);
//                 this.updateRouteInfo(data.features[0].properties, endName, endAddress, listItemId);
//             }
//         })
//         .catch(error => console.error('Error fetching TMAP route:', error));
//     },
    
//     drawRoutePolylinesWithTraffic(features) {
//         const trafficColors = { "1": "#009900", "2": "#FFD400", "3": "#FF8C00", "4": "#FF0000" };
//         features.forEach(feature => {
//             if (feature.geometry.type === "LineString") {
//                 const trafficData = feature.geometry.traffic;
//                 if (trafficData) {
//                     trafficData.forEach(([from, to, congestion]) => {
//                         const path = feature.geometry.coordinates.slice(from, to + 1).map(c => new Tmapv2.LatLng(c[1], c[0]));
//                         if (path.length < 2) return;
//                         const polyline = new Tmapv2.Polyline({ path, strokeColor: trafficColors[congestion] || "#808080", strokeWeight: 8, map: this.state.map });
//                         this.state.polylines.push(polyline);
//                     });
//                 }
//             }
//         });
//     },

//     updateRouteInfo(properties, endName, endAddress, listItemId) {
//         const listItem = document.getElementById(listItemId);
//         if (listItem && properties) {
//             const time = Math.round(properties.totalTime / 60);
//             const toll = (properties.totalFare || 0).toLocaleString('ko-KR');
//             const dist = (properties.totalDistance / 1000).toFixed(1);
//             listItem.innerHTML = `<strong>${endName}</strong><br>주소: ${endAddress}<hr>거리: ${dist}km, 시간: 약 ${time}분, 통행료: ${toll}원`;
//         }
//     },

//     // 날씨
//     getWeatherAndDisplay(lat, lon) {
//         if (!this.config.weatherApiKey) return;
//         const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${this.config.weatherApiKey}&units=metric&lang=kr`;
//         fetch(url).then(r => r.json()).then(data => {
//             if (data.list?.length > 0) {
//                 this.elements.weatherInfo.innerHTML = this.formatWeatherForecast(data);
//             }
//         }).catch(e => console.error('Weather fetch error:', e));
//     },

//     formatWeatherForecast(data) {
//         let html = `<h3>${data.city.name} 날씨 예보</h3><div class="weather-forecast-container">`;
//         const daily = {};
//         data.list.forEach(item => {
//             const day = new Date(item.dt * 1000).toLocaleString('ko-KR', { weekday: 'short' });
//             if (!daily[day]) daily[day] = { temps: [], weathers: [], icons: [] };
//             daily[day].temps.push(item.main.temp);
//             daily[day].weathers.push(item.weather[0].description);
//             daily[day].icons.push(item.weather[0].icon);
//         });

//         for (const day in daily) {
//             const d = daily[day];
//             const avgTemp = (d.temps.reduce((a, b) => a + b, 0) / d.temps.length).toFixed(1);
//             const mostCommon = arr => arr.sort((a,b) => arr.filter(v=>v===a).length - arr.filter(v=>v===b).length).pop();
            
//             // ✨ 이 HTML 구조가 CSS와 일치해야 합니다.
//             html += `<div class="weather-day">
//                 <strong class="weather-day-text">${day}</strong>
//                 <img src="http://openweathermap.org/img/w/${mostCommon(d.icons)}.png" alt="${mostCommon(d.weathers)}">
//                 <span class="weather-desc">${mostCommon(d.weathers)}</span>
//                 <span class="weather-temp">${avgTemp}°C</span>
//             </div>`;
//         }
//         return html + `</div>`;
//     },
// };
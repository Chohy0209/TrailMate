// 버튼 클릭 이벤트
document.querySelector('.start-button').addEventListener('click', function() {
    this.style.transform = 'scale(0.95)';
    setTimeout(() => {
        this.style.transform = 'translateY(-2px)';
    }, 150);
});

// 카드 호버 효과 강화
document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.background = 'rgba(255, 255, 255, 1)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.background = 'rgba(255, 255, 255, 0.95)';
    });
});

// 동적 별 생성
function createStar() {
    const star = document.createElement('div');
    star.innerHTML = '✦';
    star.className = 'star';
    const fontSize = Math.random() * 8 + 8;
    star.style.fontSize = fontSize + 'px';
    star.style.left = `calc(${Math.random() * 100}% - ${fontSize}px)`;
    star.style.top = `calc(${Math.random() * 50}% - ${fontSize}px)`;
    star.style.animationDuration = (Math.random() * 3 + 2) + 's';
    document.querySelector('.background-elements').appendChild(star);
    
    setTimeout(() => {
        star.remove();
    }, 10000);
}

// 주기적으로 별 생성
setInterval(createStar, 3000);
// static/landing_script.js
document.addEventListener('DOMContentLoaded', () => {
    // 버튼 클릭 효과
    document.querySelector('.cta-button').addEventListener('click', function() {
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.style.transform = 'translateY(-2px)';
        }, 150);
    });

    // 카드 호버 효과
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.background = 'rgba(255, 255, 255, 0.25)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.background = 'rgba(255, 255, 255, 0.1)';
        });
    });

    // 동적 별 생성
    const background = document.querySelector('.background-elements');
    function createStar() {
        if (document.querySelectorAll('.star').length > 50) return;
        const star = document.createElement('div');
        star.innerHTML = '✦';
        star.className = 'star';
        const size = Math.random() * 2 + 1;
        star.style.fontSize = `${size}vw`;
        star.style.left = `${Math.random() * 100}%`;
        star.style.top = `${Math.random() * 100}%`;
        star.style.animationDuration = `${Math.random() * 2 + 3}s`;
        background.appendChild(star);
        setTimeout(() => star.remove(), 5000);
    }
    setInterval(createStar, 1000);
});
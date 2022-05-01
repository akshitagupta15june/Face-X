const canvas1 = document.getElementById('canva1');
const ctx1 = canvas1.getContext('2d');
canvas1.width = window.innerWidth;
canvas1.height = window.innerHeight;
let particleArray1 = [];

//to handle mouse position
const mouse = {
    x: null,
    y: null,
    radius: 150
}

window.addEventListener('mousemove', function(e) {
    mouse.x = e.x+canvas1.clientLeft/2;
    mouse.y = e.y+canvas1.clientTop/2;
});

function drawImageInParticle() {
    let imageWidth = png.width;
    let imageHeight = png.height;
    const data = ctx1.getImageData(0, 0, imageWidth, imageHeight);
    ctx1.clearRect(0,0,canvas1.width,canvas1.height);

    class Particle1 {
        constructor(x, y, color) {
            this.x = x + canvas1.width/2 - png.width*2,
            this.y = y + canvas1.height/2 - png.height*2,
            this.color = color,
            this.size = 2,
            this.baseX = x + canvas1.width/2 - png.width*2,
            this.baseY = y + canvas1.height/2 - png.height*2,
            this.density = (Math.random()*10) + 2;
        }
        draw() {
            ctx1.beginPath();
            ctx1.arc(this.x, this.y, this.size, 0, Math.PI*2);
            ctx1.closePath();
            ctx1.fill();
        }
        update() {
            ctx1.fillStyle = this.color;

            //to detect collision
            let dx = mouse.x - this.x;
            let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx*dx + dy*dy);
            let forceX = dx/distance;
            let forceY = dx/distance;

            const maxDistance = 100;
            let force = (maxDistance - distance)/maxDistance;
            if(force < 0) 
                force = 0;
            
            let directionX = forceX*force*this.density*0.6;
            let directionY = forceY*force*this.density*0.6;
            
            if(distance < mouse.radius + this.size) {
                this.x -= directionX;
                this.y -= directionY;
            }
            else {
                if(this.x !== this.baseX) {
                    let dx = this.x - this.baseX;
                    this.x -= dx/20;
                }
                if(this.y !== this.baseY) {
                    let dy = this.y - this.baseY;
                    this.y -= dy/20;
                }
            }
            this.draw();
        } 
    }
    function init1() {
        particleArray1 = []

        for(let y=0,y2=data.height;y<y2;y++) {
            for(let x=0,x2=data.width;x<x2;x++) {
                if(data.data[(y*4*data.width)+(x*4)+3]>128) {
                    let positionX = x;
                    let positionY = y;
                    let color = "rgba("+data.data[(y*4*data.width)+(x*4)]+','+data.data[(y*4*data.width+(x*4)+1)]+','+data.data[(y*4*data.width+(x*4)+2)];
                    particleArray1.push(new Particle1(positionX*4,positionY*4,color));
                }
            }
        }
    }
    function animate1() {
        requestAnimationFrame(animate1);
        ctx1.fillStyle = 'rgba(0,0,0,0.05)';
        ctx1.fillRect(0, 0, innerWidth, innerHeight);
        
        for(let i=0;i<particleArray1.length;i++) {
            particleArray1[i].update();
        }
    }
    init1();
    animate1();

    window.addEventListener('resize', function() {
        canvas1.width = innerWidth;
        canvas1.height = innerHeight;
        init1();
    });
}

const png = new Image();
png.src = "data:image/png;base64,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";



window.addEventListener('mouseout',
    function() {
        mouse.x = undefined;
        mouse.y = undefined;
    }
)

window.addEventListener('load', (e) => {
    console.log('page has loaded');
    ctx1.drawImage(png,0,0);
    drawImageInParticle();
});
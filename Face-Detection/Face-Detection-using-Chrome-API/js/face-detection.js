(function() {
    if (!window.FaceDetector) return;
  
    const detectFace = () => {
      const img = document.querySelector('#targetImg');
      if (!img.getAttribute('src')) return;
      const faceDetector = new FaceDetector();
      const scale = img.width / img.naturalWidth;
      clearFaceBox();
      faceDetector
        .detect(img)
        .then(faces => faces.map(face => face.boundingBox))
        .then(faceBoxes => {
          if (faceBoxes.length === 0) alert('Sorry! No Face is detected');
          faceBoxes.forEach(faceBox => {
            const { height, width, top, left } = faceBox;
            const div = drawFaceBox(height, width, top, left, scale);
            img.parentElement.appendChild(div);
          });
        })
        .catch(err => console.log(err) || alert(err));
    };
  
    const drawFaceBox = (height, width, top, left, scale) => {
      const div = document.createElement('div');
      div.className = 'face-box';
      div.style.cssText = `
        top: ${top * scale}px;
        left: ${left * scale}px;
        height: ${height * scale}px;
        width: ${width * scale}px;
      `;
      return div;
    };
  
    const clearFaceBox = () => {
      [...document.getElementsByClassName('face-box')].forEach(e => e.remove());
    };
  
    window.detectFace = detectFace;
  })();
  

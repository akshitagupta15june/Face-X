(function() {
    const imageInput = document.querySelector('#imageInput');
    const targetImg = document.querySelector('#targetImg');
    const fileUploadBtn = document.querySelector('#fileUploadBtn');
    const notSupportedArea = document.querySelector('#not-supported-area');
    const fileReader = new FileReader();
  
    if (typeof detectFace === 'undefined') return checkPossibleReasons();
  
    fileReader.addEventListener('load', e => {
      targetImg.setAttribute('src', e.target.result);
    });
  
    targetImg.addEventListener('load', () => {
      detectFace();
    });
  
    imageInput.addEventListener('change', () => {
      const file = imageInput.files[0];
      if (file) fileReader.readAsDataURL(file);
    });
  
    function checkPossibleReasons() {
      [imageInput, targetImg, fileUploadBtn].forEach(
        e => (e.style.display = 'none')
      );
  
      let errorMsg;
      if (window.chrome) {
        errorMsg = `
            FaceDetector is an experimental feature of Chrome. 
            <br><br> 
            Turn on 'Experimental Web Platform Features' flag from 
              <br>
              <code><u>chrome://flags/#enable-experimental-web-platform-features</u></code>
              <br><br>
            and Make sure Chrome is up-to-date.
            <br><br><br>
            Hit refresh once you've done! :)
          `;
      } else {
        errorMsg = `
            FaceDetector API is only avalable in Google Chrome. 
            <br><br> 
            Use Google Chrome instead
          `;
      }
  
      notSupportedArea.innerHTML = errorMsg || 'This browser is not supported.';
      notSupportedArea.style.display = 'block';
    }
  
    fileUploadBtn.addEventListener('click', () => imageInput.click());
    targetImg.addEventListener('dblclick', () => imageInput.click());
    targetImg.addEventListener('touchend', () => imageInput.click());
    window.onresize = () => detectFace();
  })();

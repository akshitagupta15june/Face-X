#!/usr/bin/env bash

mkdir -p ./data/models/snapshots
cd data/models/snapshots || exit

# Unsupervised Autoencoder
wget -O tmp.zip https://www.dropbox.com/sh/gfb0q5da8ydl8e4/AABQ6834sBvo-9_SDabVwRLra
unzip tmp.zip -d aae
rm tmp.zip

# Demo model
wget -O tmp.zip https://www.dropbox.com/sh/wzi69cwksf8iaxf/AAC0lsqTXEyRfaC9nWP6hlCMa
unzip tmp.zip -d demo
rm tmp.zip

# Landmarks trained on WFLW
wget -O tmp.zip https://www.dropbox.com/sh/n9zija0hk68g499/AACJ0pciymwdi4WtY2sX2i9Wa
unzip tmp.zip -d lms_wflw
rm tmp.zip

# Landmarks trained on 300W
wget -O tmp.zip https://www.dropbox.com/sh/u1jnl93fl7wstvt/AAAVJfVGotz0KbQtklfWkGjma
unzip tmp.zip -d lms_300w
rm tmp.zip

# Landmarks trained on AFLW
https://www.dropbox.com/sh/ayhghbeqcj3qbb2/AAAP6Jy2FEZv_rhq5T3zxnOza
unzip tmp.zip -d lms_aflw
rm tmp.zip

const Animation = require('Animation');
const Blocks = require('Blocks');
const Patches = require('Patches');
const Reactive = require('Reactive');
const Scene = require('Scene');

export class JumpGame {

    constructor() {

        this.initialize();
    }

    async initialize() {

        this.score = Reactive.add(0, 0); // cast score as reactive signal

        // fetch settings
        [
            this.playerContainer,
            this.platformContainer,
            this.playerFollow

        ] = await Promise.all([
            Scene.root.findFirst('player'),
            Scene.root.findFirst('platforms'),
            Scene.root.findFirst('playerFollow'),
            this.initGameSettings(),
            this.initPlayerSettings(),
            this.initPlatformSettings(),
        ]);

        this.playerFollow.transform.y = 0; // reset player follow y

        // setup player and platforms
        await this.setupPlayer();
        await this.setupPlatforms();

        this.fall(); // make player fall
        this.playerBlock.hidden = false; // show player
    }

    async initGameSettings() {

        // fetch game settings scene object
        let settingsObj = await Scene.root.findFirst('gameSettings');

        // get game settings
        [
            this.gameSize,
            this.screenDimensions,
            this.platformRows,

        ] = await Promise.all([
            settingsObj.outputs.getPoint2D('Game Size'),
            settingsObj.outputs.getPoint2D('Screen Dimensions'),
            settingsObj.outputs.getScalar('Platforms'),
        ]);

        // adjust settings
        this.screenDimensionsX = this.screenDimensions.x.pinLastValue();
        this.screenDimensionsY = this.screenDimensions.y.pinLastValue();
        this.platformsHeight = this.screenDimensionsY * 1.25;
        this.platformOffset = this.platformsHeight - this.screenDimensionsY;
        this.gameSizeX = this.gameSize.x.pinLastValue();
        this.gameSizeY = this.gameSize.y.pinLastValue();
        this.gameScale = this.screenDimensionsX / this.gameSizeX;
        this.platformsNum = this.platformRows.pinLastValue();
    }

    async initPlayerSettings() {

        // fetch player settings scene object
        let settingsObj = await Scene.root.findFirst('playerSettings');

        // get player settings
        [
            this.playerInputX,
            this.playerSize,
            this.playerShowHitbox,
            this.playerHitboxOffset,
            this.playerHitboxSize,
            this.playerJump,
            this.playerFall,
            this.playerBoost,
            this.playerJumpHeight,
            this.playerJumpTime,
            this.playerBoostHeight,
            this.playerMaxY

        ] = await Promise.all([
            settingsObj.outputs.getScalar('Input X'),
            settingsObj.outputs.getPoint2D('Size'),
            settingsObj.outputs.getBoolean('Show Hitbox'),
            settingsObj.outputs.getPoint2D('Hitbox Offset'),
            settingsObj.outputs.getPoint2D('Hitbox Size'),
            settingsObj.outputs.getShader('Jump'),
            settingsObj.outputs.getShader('Fall'),
            settingsObj.outputs.getShader('Boost'),
            settingsObj.outputs.getScalar('Jump Height'),
            settingsObj.outputs.getScalar('Jump Time'),
            settingsObj.outputs.getScalar('Boost Height'),
            settingsObj.outputs.getScalar('Max Y'),
        ]);

        // scale player responsively
        this.playerSize = this.playerSize.mul(this.gameScale);
        this.playerHitboxOffset = this.playerHitboxOffset.mul(this.gameScale);
        this.playerHitboxSize = this.playerHitboxSize.mul(this.gameScale);
        this.playerJumpHeight = this.playerJumpHeight.mul(this.gameScale).pinLastValue();
        this.playerJumpTime = this.playerJumpTime.pinLastValue() * 1000;
        this.playerBoostHeight = this.playerBoostHeight.mul(this.gameScale).pinLastValue();

        // calculate gravity
        this.gravity = (2 * this.playerJumpHeight) / (this.playerJumpTime * this.playerJumpTime);

        // calculate how long boost should last
        this.playerBoostTime = Math.floor(Math.sqrt((2 * this.playerBoostHeight) / this.gravity))

        // calculate how long fall should last
        this.playerFallTime = Math.floor(Math.sqrt((2 * this.screenDimensionsY) / this.gravity))
    }

    async initPlatformSettings() {

        // fetch platform settings scene object
        let settingsObj = await Scene.root.findFirst('platformSettings');

        // get platform settings
        [
            this.platformSize,
            this.platformShowHitbox,
            this.platformHitboxOffset,
            this.platformHitboxSize,
            this.platformNormal,
            this.platformBoost,
            this.platformFake

        ] = await Promise.all([
            settingsObj.outputs.getPoint2D('Size'),
            settingsObj.outputs.getBoolean('Show Hitbox'),
            settingsObj.outputs.getPoint2D('Hitbox Offset'),
            settingsObj.outputs.getPoint2D('Hitbox Size'),
            settingsObj.outputs.getShader('Normal Platform'),
            settingsObj.outputs.getShader('Boost Platform'),
            settingsObj.outputs.getShader('Fake Platform'),
        ]);

        // scale platform responsively
        this.platformSize = this.platformSize.mul(this.gameScale);
        this.platformHitboxOffset = this.platformHitboxOffset.mul(this.gameScale);
        this.platformHitboxSize = this.platformHitboxSize.mul(this.gameScale);
    }

    async setupPlatforms() {

        // loop through the platforms and set each one up
        let queue = [];

        for (var i = 0; i < this.platformsNum; i++) {
            queue.push(this.setupPlatform(i));
        }

        await Promise.all(queue); // wait for all to setup 
    }

    async setupPlatform(index) {

        // initialize platforms array if not already
        this.platforms = this.platforms || [];

        let platformChances = {
            0: 0.8, // 60% chance normal platform
            1: 0.1, // 20% chance boost platform
            2: 0.1 // 20% chance fake platform
        };

        // get platform kind normal/boost/fake
        let platformKind = this.weightedRnd(platformChances);

        // instantiate the a platform block
        let p = await Blocks.instantiate('platform');

        // platform x starting position is random
        let posX = this.rndInt(this.screenDimensionsX - this.platformSize.x.pinLastValue());

        // platform y position calculated from index
        let posY = (this.platformsHeight / this.platformsNum) * index;

        // offset platform y so they spawn above the screen
        posY = Reactive.add(posY - this.platformOffset, 0);

        // 1st platform should appear underneath player
        if (index === this.platformsNum - 1) {
            posX = this.screenDimensionsX * 0.5 - (this.platformSize.x.pinLastValue() * 0.5);
            platformKind = 0;
        }
        // 2nd platform out of way on left
        if (index === this.platformsNum - 2) {
            posX = 0;
        }
        // 3rd platform out of way on right
        if (index === this.platformsNum - 3) {
            posX = this.screenDimensionsX - this.platformSize.x.pinLastValue();
        }

        // keep track of position
        p.platformX = posX;
        p.platformY = posY;

        // populate platform settings
        p.inputs.setPulse('Reset', Reactive.once());
        p.inputs.setBoolean('Enable', true);
        p.inputs.setShader('Texture Normal', this.platformNormal);
        p.inputs.setShader('Texture Boost', this.platformBoost);
        p.inputs.setShader('Texture Fake', this.platformFake);
        p.inputs.setPoint2D('Position', Reactive.pack2(posX, posY));
        p.inputs.setPoint2D('Size', this.platformSize);
        p.inputs.setBoolean('Show Hitbox', this.platformShowHitbox);
        p.inputs.setPoint2D('Hitbox Offset', this.platformHitboxOffset);
        p.inputs.setPoint2D('Hitbox Size', this.platformHitboxSize);
        p.inputs.setPoint2D('Collide With Position', this.playerPos.add(this.playerHitboxOffset));
        p.inputs.setPoint2D('Collide With Size', this.playerHitboxSize);
        p.inputs.setBoolean('Normal', platformKind === 0);
        p.inputs.setBoolean('Boost', platformKind === 1);
        p.inputs.setBoolean('Fake', platformKind === 2);
        

        // add platform to scene
        await this.platformContainer.addChild(p);

        // listen to collisions
        let hit = await p.outputs.getPulse('Hit');

        hit.subscribe(() => {

            // hit normal platform
            if (platformKind === 0) {

                // set player to jump state
                this.playerBlock.inputs.setPulse('Jump', Reactive.once());

                // make player jump
                this.jump(this.playerJumpHeight, this.playerJumpTime);
            }

            // hit boost platform
            if (platformKind === 1) {

                // set player to boost state
                this.playerBlock.inputs.setPulse('Boost', Reactive.once());

                // make player boost jump
                this.jump(this.playerBoostHeight, this.playerBoostTime);
            }
        });

        // monitor platform position
        let platformPos = await p.outputs.getPoint2D('Position');

        platformPos.y.lt(0).monitor().subscribe(function(e) {

            // if platform is above screen
            if (e.newValue === true) {

                // select new random platform kind
                platformKind = this.weightedRnd(platformChances);

                // set platform kind
                p.inputs.setBoolean('Normal', platformKind === 0);
                p.inputs.setBoolean('Boost', platformKind === 1);
                p.inputs.setBoolean('Fake', platformKind === 2);

                // reset platform
                p.inputs.setPulse('Reset', Reactive.once());
            }
        }.bind(this))

        // add to list of platforms
        this.platforms.push(p);
    }

    async setupPlayer() {

        // instantiate player block
        this.playerBlock = await Blocks.instantiate('player');

        // populate player settings        
        this.playerBlock.inputs.setShader('Jump Texture', this.playerJump);
        this.playerBlock.inputs.setShader('Boost Texture', this.playerBoost);
        this.playerBlock.inputs.setShader('Fall Texture', this.playerFall);
        this.playerBlock.inputs.setPoint2D('Size', this.playerSize);
        this.playerBlock.inputs.setBoolean('Show Hitbox', this.playerShowHitbox);
        this.playerBlock.inputs.setPoint2D('Hitbox Offset', this.playerHitboxOffset);
        this.playerBlock.inputs.setPoint2D('Hitbox Size', this.playerHitboxSize);

        // hide player block until ready
        this.playerBlock.hidden = true;

        // add block to scene
        await this.playerContainer.addChild(this.playerBlock);

        // player X follows player input X (nose tip)
        this.playerFollow.transform.x = this.playerInputX.sub(this.playerSize.x.mul(0.5));
        // player Y start position 60% vertical
        this.playerFollow.transform.y = this.screenDimensionsY * 0.6 - (this.playerSize.y.pinLastValue() * 0.5);

        // create vector2 for tracking player position
        this.playerPos = Reactive.pack2(
            this.playerFollow.transform.x,
            this.playerFollow.transform.y
        );

        // monitor player Y, if falls off screen is game over
        this.playerFollow.transform.y.gt(this.platformsHeight).monitor().subscribe(function(e) {

            // if player off screen
            if (e.newValue === true) {
                this.gameOver();
            }
        }.bind(this));

        // set player position
        this.playerBlock.inputs.setPoint2D('Position', this.playerPos);
    }

    async jump(jumpHeight, duration) {

        const from = this.playerFollow.transform.y.pinLastValue(); // previous y position
        const to = from - jumpHeight; // previous y position + jump height

        // new timer driver
        const timeDriver = Animation.timeDriver({ durationMilliseconds: duration });

        // quadratic out easing to simulate gravity
        const sampler = Animation.samplers.easeOutQuad(from, to);

        // animate y position
        const playerY = Animation.animate(timeDriver, sampler);

        // clamp player position and calculate platform offset 
        // (makes platforms appear to travel down as you travel upwards)
        const platformOffset = Reactive.sub(this.playerMaxY, playerY).max(0);

        // set player Y position
        this.playerFollow.transform.y = playerY.max(this.playerMaxY);

        timeDriver.start(); // Start the time driver

        // after jump, player falls
        timeDriver.onCompleted().subscribe(function() {
            this.fall();
        }.bind(this))

        // configure platforms
        for (const p of this.platforms) {

            p.inputs.setBoolean('Enable', false); // disable colliders while jumping

            // set platform y position
            p.platformY = platformOffset
                .add(p.platformY.pinLastValue() + this.platformOffset)
                .mod(this.platformsHeight) //(if platform goes off screen move back to top of screen)
                .sub(this.platformOffset);

            // random platform x position if offscreen
            p.platformX = Reactive.ifThenElse(
                p.platformY.lt(0),
                this.rndInt(this.screenDimensionsX - this.platformSize.x.pinLastValue()),
                p.platformX);

            // set platform position
            p.inputs.setPoint2D('Position', Reactive.pack2(
                p.platformX,
                p.platformY
            ))
        }

        // score is the total platform offset (total height reached)
        this.score = platformOffset
            .add(this.score.pinLastValue())
            .floor();

        // update score
        Patches.inputs.setScalar('OUTPUT_SCORE', this.score);
    }

    async fall() {

        // set fall state on player object
        this.playerBlock.inputs.setPulse('Fall', Reactive.once());

        const from = this.playerFollow.transform.y.pinLastValue(); // previous y position
        const to = from + this.screenDimensionsY; // previous y position + screen height

        // new timer driver
        const timeDriver = Animation.timeDriver({ durationMilliseconds: this.playerFallTime });

        // quadratic in easing to simulate gravity
        const sampler = Animation.samplers.easeInQuad(from, to);

        // set player Y position
        this.playerFollow.transform.y = Animation.animate(timeDriver, sampler);

        timeDriver.start(); // Start the time driver

        // enable platform collisions while falling
        for (const p of this.platforms) {
            p.inputs.setBoolean('Enable', true)
        }
    }

    gameOver() {

        // set player state die
        this.playerBlock.inputs.setPulse('Die', Reactive.once());

        // trigger game over pulse
        Patches.inputs.setPulse('GAME_OVER', Reactive.once());
    }

    // get a random integer
    rndInt(max) {
        return Math.floor(Math.random() * max);
    }

    // get a random number based of a weighted probability
    weightedRnd(spec) {

        let i;
        let sum = 0;
        let r = Math.random();

        for (i in spec) {
            sum += spec[i];
            if (r <= sum) return Math.floor(i);
        }
    }
};

export const jg = new JumpGame();
const Patches = require('Patches');
const Persistence = require('Persistence');
const Reactive = require('Reactive');

export class Score {

    constructor() {

        this.initialize();
    }

    async initialize() {

        // start score is zero
        this.score = 0;
        this.highScore = await this.getHighScore();

        // fetch add/lose point amounts
        this.score = await Patches.outputs.getScalar('SCORE');

        // listen to save high score pulse
        let saveScore = await Patches.outputs.getPulse('SAVE_SCORE');
        saveScore.subscribe(() => this.onSaveScore());

        this.updateScore();
    }

    onSaveScore() {

        this.saveHighScore(); // save the high score (if applicable)
    }

    async getHighScore() {

        let highScore = 0;

        // Store a reference to the userScope
        const userScope = Persistence.userScope;

        try {
            // Fetch saved data (if any)
            const result = await userScope.get('score');

            // Check if we have a previously saved high score
            if (result !== null &&
                typeof result.highScore !== 'undefined') {

                // found a saved high score
                highScore = result.highScore;
            }
        } catch (ex) {
            // console.log(ex);
        }

        return highScore;
    }

    async saveHighScore() {

    	let score = this.score.pinLastValue();

    	this.updateScore(score);

        if (score < this.highScore) {
            return; // score wasnt a high score, bail
        }

        // This score is the new high score
        this.highScore = score;

        // Create a JavaScript object to store the data
        const data = { highScore: this.highScore };

        // Store a reference to the userScope
        const userScope = Persistence.userScope;


        try {
            // Store the data
            await userScope.set('score', data);
        } catch (ex) {
            // console.log(ex);
        }
    }

    updateScore() {

    	let score = this.score.pinLastValue();

    	// show current high score
        let gameOverText = `high score ${this.highScore}`;


        // check for new high score
        if(score > this.highScore) {

        	gameOverText = 'new high score';
        	// Send pulse that we have a new high score
        	Patches.inputs.setPulse('NEW_HIGH_SCORE', Reactive.once());
        }

        // send game over text
        Patches.inputs.setString('GAME_OVER_TEXT', gameOverText);
    }
};

export const s = new Score();
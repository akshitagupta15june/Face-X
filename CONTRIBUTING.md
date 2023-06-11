# Contributing Guidelines

This documentation contains a set of guidelines to help you during the contribution process.

## Submitting Contributions👩‍📈

Below you will find the process and workflow used to review and merge your changes.

1. Find/Create an issue

- Take a look at the Existing Issues or create your **own** Issues!
- Wait for the Issue to be assigned to you after which you can start working on it.
- Note : Every change in this project should/must have an associated issue.


2. Fork the repository by clicking on the ["Fork"](https://github.com/akshitagupta15june/Face-X/fork) button at the top right corner of the page.


3. Once you are in your fork the url should look something like
`https://github.com/<YOUR-USERNAME>/Face-X/`

4. Go to preferred folder in your computer and paste the following command (Only one of it if you don't have ssh setup then go with HTTP command)
- For HTTP
 ```
   git clone https://github.com/<YOUR-USERNAME>/Face-X.git
 ```
- For SSH
 ```bash
   git clone git@github.com:<YOUR-USERNAME>/Face-X.git
 ```

5. Now enter the folder by running the following command
```bash
cd Face-X
```

5. Now you are in the `Face-X` folder

6. Now go ahead and create a new branch, then switch to that branch.
```bash
git checkout -b fix-issue-<ISSUE-NUMBER>
```

7.  Make your changes to the codebase. Test your changes to make sure they work as expected.

8. After done you can now push this changes. for doing that follow the following command chain
   - `git status` (Shows the changed files)
   - `git add .` (Will add all the files to staging area)
   - `git commit -m "feat/docs/fix: :emoji-name: <EXPLAIN-YOUR_CHANGES>"`
   - `git push origin fix-issue-<ISSUE-NUMBER>`

- **NOTE**: A PR should have only one commit. Multiple commits should be squashed.

9.  Create a Pull Request:
- Go to your forked GitHub repository and navigate to the Pull Request section.
- Click on the pop-up that says "Pull Request" to be redirected to the pull request page.
- Fill in the form template of the pull request, adding a title and description that explain your contribution.
- Include screenshots if they help clarify the enhancement/implementation/bug/documentation.
- Click on **Submit** to create the pull request*

10. Hurrey! You just did your contribution to this project 🎉

11. Wait for your pull request to be reviewed and merged.

>NOTE: Please make sure to follow the [Code of conduct](https://github.com/akshitagupta15june/Face-X/blob/master/CODE_OF_CONDUCT.md) while contributing.


## Alternatively, using GitHub Codespaces:

1. Follow steps 1 to 3 above, to fork the repository.
2. Navigate to the forked repository i.e. `https://github.com/<YOUR-USERNAME>/Face-X/` 
3. Click **Code**, then click **Create codespace on master**.

    <img src='https://github.com/akshitagupta15june/Face-X/tree/master/.github/user/codespace_initial.jpeg' style="max-width: 50%; height: auto;">

4. A new codespace will be created. 
5. Follow steps 6 to 9 above using the terminal in the codespace. Instead of step 9 you can also directly create a PR through codespace by going to the **Source Control view** in **Activity Bar**.
     
     <img src='https://github.com/akshitagupta15june/Face-X/tree/master/.github/user/codespace_PR.jpeg' style="max-width: 50%; height: auto;">


## Alternatively, using GitHub Desktop: 
1. Open GitHub Desktop and log in to your GitHub account.

2. Make sure you are on the "Current Repository" view. If not, go to "File" and select "Add Local Repository" to add your repository.

3. In the "Current Repository" view, ensure you are on the branch that you want to submit a pull request for. If you're not on the correct branch, use the "Branch" menu to switch to the correct branch.

4. Once you're on the correct branch, make your changes and commit them to the branch. You can do this by clicking the "+" button in the upper-left corner of the GitHub Desktop window, making your changes, and then entering a commit message.

5. After you've made your changes and committed them, click the "Push origin" button in the top-right corner of the GitHub Desktop window. This will push your changes to the remote repository on GitHub.

6. Now, go to the GitHub website, navigate to your fork of the repository, and you should see a button to "Compare & pull request" between your fork and the original repository, click on it.

7. On the pull request page, you can review your changes and add any additional information, such as a title and a description, that you want to include with your pull request.

8. Once you're satisfied with your pull request, click the "Create pull request" button to submit it.

**Note:** In order to create a pull request, you must have a fork of the original repository in your GitHub account and you must have made the changes in that forked repository.


## Your Pull Request has been submitted and will be reviewed by the maintainer and merged.



### Need more help?🤔

You can refer to the following articles on basics of Git and Github and also contact the Project Mentors,
in case you are stuck:

- [Watch this video to get started, if you have no clue about open source](https://youtu.be/SYtPC9tHYyQ)
- [Forking a Repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
- [Cloning a Repo](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request)
- [How to create a Pull Request](https://opensource.com/article/19/7/create-pull-request-github)
- [Getting started with Git and GitHub](https://towardsdatascience.com/getting-started-with-git-and-github-6fcd0f2d4ac6)
- [Learn GitHub from Scratch](https://lab.github.com/githubtraining/introduction-to-github)


Hope you will learn something new while contributing in this project!!😇😇😇😇

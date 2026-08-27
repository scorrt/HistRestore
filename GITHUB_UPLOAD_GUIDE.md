# Simple GitHub Upload Guide

This is the easiest manual upload route.

## Option A: GitHub Web Upload

1. Go to <https://github.com/new>.
2. Create a repository, for example:

   ```text
   HistRestore
   ```

3. Choose `Private` first. After checking the repository, you can switch it to `Public`.
4. Do not initialize with README, license, or `.gitignore`, because this folder already contains them.
5. Open the new repository page and click **uploading an existing file**.
6. Drag all files and folders from:

   ```text
   HistRestore-public-release/
   ```

7. Commit with a message such as:

   ```text
   Initial sanitized reproducibility release
   ```

8. After upload, search the GitHub repository for private artifacts, including:

   ```text
   credentials
   server addresses
   local absolute paths
   private dataset links
   model weights
   ```

   If anything private appears, keep the repo private and remove it before public release.

## Option B: GitHub Desktop

1. Install and log into GitHub Desktop.
2. Choose **File -> Add local repository**.
3. Select:

   ```text
   HistRestore-public-release/
   ```

4. If prompted, choose **create a repository**.
5. Commit all files.
6. Click **Publish repository**.
7. Start as `Private`, then switch to `Public` after checking.

## Recommended MDPI Link Timing

For submission, either:

- keep the repository private and write `available upon acceptance`, or
- publish a public repository and optionally archive a release on Zenodo to get a DOI.

The cleanest final statement is:

```text
Code and split manifests are available at: https://github.com/<user>/HistRestore
```

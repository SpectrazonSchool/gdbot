# Changelog

## v1.4.1
- Now runs on macOS, Android, and iOS in addition to Windows
- Fixed a non-portable timestamp call that broke building outside Windows

## v1.4.0
- Training no longer auto-plays its showcase, it pauses and asks first
- Runs are now saved either way, so overnight runs aren't wasted
- New playback library: save, replay, rename, and delete best runs per level

## v1.3.2
- New Max FPS setting to break the speed ceiling
- Speed now actually pays off at high frame rates, especially with Hide gfx on

## v1.3.1
- Less slowdown over long training runs
- Genomes now cap dead connections instead of bloating
- Innovation lookup is now a hash map instead of a linear scan
- Smaller settings info buttons
- Moved the Hide gfx info button so it no longer overlaps text

## v1.3.0
- Major bug fixes and learning optimisations
- Mutations only take control near where the parent died
- Passed sections can no longer be lost to a bad mutation
- Randomized takeover point for better failed-jump fixes

## v1.2.0
- Revamped the entire training system to rely less on approximations and more on using the actual GD game loop
- Training scores now reproduce exactly on the showcase run
- Removed the Batch setting (attempts are sequential now)
- Default Speed raised from 8x to 16x

## v1.1.1
- Your real player icon now plays as the current best genome during training
- Ghost players get distinct semi-transparent colors

## v1.1.0
- Ghosts now shrink and grow through mini portals
- Ghosts stay matched to the real player's speed and gamemode
- Fixed a fitness-sharing bug that stalled the AI at the first obstacle
- Added elitism so the best run is never lost
- AI can now learn to wait through "do nothing" sections
- Info buttons on every training setting
- Added an about page

## v1.0.0
- Initial release

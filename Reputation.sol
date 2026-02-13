// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Reputation {
    uint256 constant THRESHOLD = 50;        // آستانه میانگین امتیاز (۵۰٪)
    uint256 constant HISTORY_LENGTH = 5;    // تعداد راندهای اخیر برای تصمیم‌گیری

    struct Participant {
        uint256[] recentScores;   // آخرین HISTORY_LENGTH امتیاز
        bool blacklisted;         // وضعیت تحریم
    }

    mapping(address => Participant) public participants;
    address public owner;

    event ScoreUpdated(address indexed participant, uint256 score);
    event Blacklisted(address indexed participant);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only server can update scores");
        _;
    }

    // ثبت امتیاز جدید و بررسی خودکار شرط تحریم
    function updateScore(address participant, uint256 score) external onlyOwner {
        require(score <= 100, "Score must be 0-100");
        Participant storage p = participants[participant];

        // افزودن به تاریخچه
        p.recentScores.push(score);
        if (p.recentScores.length > HISTORY_LENGTH) {
            // حذف قدیمی‌ترین عنصر (روش شیفت)
            for (uint i = 0; i < p.recentScores.length - 1; i++) {
                p.recentScores[i] = p.recentScores[i + 1];
            }
            p.recentScores.pop();
        }

        emit ScoreUpdated(participant, score);

        // اگر هنوز تحریم نشده و HISTORY_LENGTH امتیاز جمع شده، میانگین را حساب کن
        if (!p.blacklisted && p.recentScores.length == HISTORY_LENGTH) {
            uint256 sum = 0;
            for (uint i = 0; i < HISTORY_LENGTH; i++) {
                sum += p.recentScores[i];
            }
            uint256 avg = sum / HISTORY_LENGTH;
            if (avg < THRESHOLD) {
                p.blacklisted = true;
                emit Blacklisted(participant);
            }
        }
    }

    // مشاهده وضعیت تحریم
    function isBlacklisted(address participant) external view returns (bool) {
        return participants[participant].blacklisted;
    }

    // مشاهده تاریخچه امتیازات
    function getRecentScores(address participant) external view returns (uint256[] memory) {
        return participants[participant].recentScores;
    }
}
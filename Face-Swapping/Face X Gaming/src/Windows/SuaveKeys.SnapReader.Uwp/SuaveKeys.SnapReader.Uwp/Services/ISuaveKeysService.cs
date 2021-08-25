using ServiceResult;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace SuaveKeys.SnapReader.Uwp.Services
{
    public interface ISuaveKeysService
    {
        Task<Result<bool>> StartSignInAsync();
        Task<Result<string>> GetAccessTokenAsync();
        Task<Result<bool>> SendCommandAsync(string command);
    }
}
